from typing import Literal

from typing_extensions import TypedDict


class ToolCallResult(TypedDict):
    """
    A dictionary representing the result of a tool call.

    Attributes:
        tool_name (str): The name of the tool that was called.
        tool_input (str): The input provided to the tool.
        tool_output (str): The output returned by the tool.
    """

    llm_consumable: str
    ui_displayable: str
    output_type: Literal["dataframe"]
