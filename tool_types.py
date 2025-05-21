from typing import Any, Literal

from typing_extensions import TypedDict


class ToolCallResult(TypedDict):
    """
    A TypedDict representing the result of a tool call.
    Attributes:
        llm_consumable (str): A string representation of the result that can be consumed by a language model.
        ui_displayable (Any): Any type of data that can be displayed in a user interface.
        return_type (Literal["dataframe", "error_message"]): The type of the returned data, either a dataframe or an error message.
    """

    llm_consumable: str
    ui_displayable: Any
    return_type: Literal["dataframe", "error_message"]
