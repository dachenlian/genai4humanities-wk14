import json
import os
from collections.abc import AsyncGenerator
from copy import deepcopy
from time import time

import gradio as gr
import lancedb
import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._generated.types import ChatCompletionOutputToolCall
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from chat_types import Message
from tool_types import ToolCallResult
from utils import create_tool_schema_for_function

load_dotenv()

API_KEY = os.getenv("HF_TOKEN", "")


embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu",
)
ds = load_from_disk("./data/dataset_processed/")
df = ds.to_pandas()
db = lancedb.connect("./data/lance_db")
tbl = db.create_table("movies", data=df, mode="overwrite")
tbl.create_fts_index("overview", replace=True)

COLUMNS_TO_KEEP = ["title", "overview", "release_year", "cast"]


class QueryMovieDB(BaseModel):
    text: str = Field(
        description="Query overviews of movies",
    )
    limit: int = Field(
        default=5,
        description="Number of results to return",
    )


def query_movie_db(
    text: str,
    limit: int = 5,
) -> ToolCallResult:
    """
    Query the LanceDB movie database for movies with similar overviews to the input text.

    Args:
        text (str): The input text to query the database.
        limit (int, optional): The number of results to return. Defaults to 5.

    Returns:
        ToolCallResult: The result of the tool call.
    """
    q_emb = embedder.encode(text)
    df = (
        # tbl.search(q_emb).limit(limit).to_pandas().drop(columns=["vector", "_distance"])
        tbl.search(q_emb).limit(limit).to_pandas()[COLUMNS_TO_KEEP]
    )
    df["cast"] = df["cast"].apply(lambda x: [c["name"] for c in x])
    df = df.sort_values(by="release_year", ascending=False)
    return {
        "llm_consumable": df.to_json(lines=True, orient="records"),
        "ui_displayable": df,
        "return_type": "dataframe",
    }


TOOLS = [
    (query_movie_db, QueryMovieDB),
]

AVAILABLE_FUNCTIONS = {
    "query_movie_db": query_movie_db,
}

tool_schemas = [create_tool_schema_for_function(fn, schema) for fn, schema in TOOLS]

CURRENT_DF = pd.DataFrame(columns=COLUMNS_TO_KEEP)


def call_function(name, args) -> ToolCallResult:
    func = AVAILABLE_FUNCTIONS.get(name)
    if not func:
        raise ValueError(f"Unknown function: {name}")
    try:
        # Call the function with the provided arguments
        return func(**args)
    except TypeError as e:  # Catches errors like missing/extra arguments
        error_msg = f"Error: Argument mismatch when calling tool '{name}' with arguments {args}. Details: {e}"
        print(error_msg)
        return {
            "llm_consumable": error_msg,
            "ui_displayable": error_msg,
            "return_type": "error_message",
        }
    except Exception as e:  # Catches other errors during tool execution
        error_msg = f"Error during execution of tool '{name}' with arguments {args}. Details: {e}"
        print(error_msg)
        return {
            "llm_consumable": error_msg,
            "ui_displayable": error_msg,
            "return_type": "error_message",
        }


gr.Info(f"Loaded movie database with {tbl.count_rows()} rows")


# https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance
# https://www.gradio.app/guides/creating-a-chatbot-fast
async def chat(
    message: str,
    history: list[Message],
    api_key: str = API_KEY,
    provider: str = "hf-inference",
    model: str = "Qwen/Qwen3-235B-A22B",
) -> AsyncGenerator[tuple[list[gr.ChatMessage] | gr.ChatMessage, pd.DataFrame], None]:
    global CURRENT_DF
    start_time = time()
    if not api_key:
        raise gr.Error("Hugging Face API key is required")
    if not message:
        raise gr.Error("Message cannot be empty")

    client = AsyncInferenceClient(provider=provider, api_key=api_key)
    messages = deepcopy(history)
    messages.append(
        {"role": "user", "content": f"{message} /no_think"}
    )  # /no_think disables thinking
    response = await client.chat_completion(  # type: ignore
        model=model,
        messages=messages,
        tools=tool_schemas,
        tool_choice="auto",
        # stream=True,  # streaming is a lot of work to handle tool calls and regular messages
    )  # type: ignore
    logger.info(f"Response: {response}")

    responses: list[gr.ChatMessage] = []

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        tc: ChatCompletionOutputToolCall = tool_calls[0]
        logger.info(f"Tool call: {tc}")

        call_id = tc.id
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        # responses.append(gr.ChatMessage(
        #     role="assistant",
        #     content=f"Tool call: {tc.function.name}({', '.join(tc.function.arguments.values())})",
        # ))
        tool_result = call_function(func_name, func_args)
        if tool_result["return_type"] == "dataframe":
            CURRENT_DF = tool_result["ui_displayable"]
        logger.info(f"Tool result: {tool_result['llm_consumable']}")
        meta_response = gr.ChatMessage(
            content="",
            metadata={
                "title": f"Tool call: {func_name}",
                "status": "pending",
            },
        )
        yield meta_response, CURRENT_DF
        messages.append(
            {
                "role": "tool",
                "content": tool_result["llm_consumable"],
                # "content": f"Tool call: {func_name}({', '.join(func_args.values())})",
            }
        )
        response = await client.chat_completion(  # type: ignore
            model=model,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            # stream=True,  # streaming is a lot of work to handle tool calls and regular messages
        )
        logger.info(f"Response after tool call: {response}")
        meta_response.metadata["status"] = "done"
        meta_response.metadata["duration"] = time() - start_time
        yield meta_response, CURRENT_DF

        responses.append(meta_response)
    responses.append(
        gr.ChatMessage(
            role="assistant",
            content=response.choices[0].message.content,
        )
    )
    yield responses, CURRENT_DF


with gr.Blocks() as demo:
    df_component = gr.Dataframe(
        CURRENT_DF,
        label="Movie Search Results",
        render=False,
        headers=["Title", "Overview", "Release Year", "Cast"],
        datatype=["str", "str", "number", "str"],
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.ChatInterface(
                fn=chat,
                type="messages",
                title="Movie Search",
                theme=gr.themes.Default(),  # https://www.gradio.app/guides/theming-guide
                additional_inputs=[
                    gr.Textbox(
                        label="Hugging Face API Key",
                        placeholder="Enter your Hugging Face API key here",
                        type="password",
                        value=API_KEY,
                    ),
                    gr.Dropdown(
                        label="Provider",
                        choices=["hf-inference"],
                        value="hf-inference",
                    ),
                    gr.Dropdown(
                        label="Model",
                        choices=["Qwen/Qwen3-235B-A22B"],
                        value="Qwen/Qwen3-235B-A22B",
                    ),
                ],
                additional_outputs=[df_component],
            )
    with gr.Row():
        with gr.Column(scale=1):
            df_component.render()


# if __name__ == "__main__":
demo.launch(pwa=True, share=True)
