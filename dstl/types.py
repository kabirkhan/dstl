from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator


class Task(str, Enum):
    NER = "NER"


class Span(BaseModel):
    """Entity Span in Example"""

    text: str
    start: int
    end: int
    label: str
    token_start: Optional[int]
    token_end: Optional[int]


class Token(BaseModel):
    """Token with offsets into Example Text"""

    text: str
    start: int
    end: int
    id: int


class Example(BaseModel):
    """Example with NER Label spans"""

    text: str
    spans: List[Span]
    tokens: Optional[List[Token]]
    meta: Dict[str, Any] = {}
    formatted: bool = False

    @root_validator(pre=True)
    def span_text_must_exist(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("formatted", False):
            # Ensure each span has a text property
            spans = values["spans"]
            for span in spans:
                if not isinstance(span, Span):
                    if "text" not in span:
                        span["text"] = values["text"][span["start"] : span["end"]]

            # Ensure the meta has a source property
            # if something that's not a dict is passed in
            meta = values.get("meta", {})
            if isinstance(meta, list) or isinstance(meta, str):
                meta = {"source": meta}

            values["spans"] = spans
            values["meta"] = meta
            values["formatted"] = True

        return values
