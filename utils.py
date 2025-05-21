from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from typing_extensions import TypedDict

FunctionSchema = TypedDict(
    "FunctionSchema",
    {
        "name": str,
        "description": str,
        "parameters": dict[str, Any],
    },
)


ToolSchema = TypedDict(
    "ToolSchema",
    {
        "type": str,
        "function": FunctionSchema,
    },
)


def create_tool_schema_for_function(
    func: Callable, params_model_class: type[BaseModel]
) -> ToolSchema:
    """
    Generates an OpenAI-compatible tool schema for a given function and its Pydantic parameter model.
    """
    if not func.__doc__:
        raise ValueError(
            f"Function {func.__name__} must have a docstring for its description."
        )

    # Pydantic V2's method to get JSON schema
    parameters_schema = params_model_class.model_json_schema()

    tool_schema: ToolSchema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip().split("\n")[
                0
            ],  # Use the first line of the docstring
            "parameters": parameters_schema,
        },
    }
    return tool_schema
