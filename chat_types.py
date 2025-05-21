from typing_extensions import TypedDict


class Message(TypedDict):
    """
    A TypedDict representing a message in a chat interface.
    """

    role: str
    content: str
