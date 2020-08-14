from dstl.translate.core import match_example
from dstl.types import Example, Span, Token


def test_match_example():
    lang = "es"
    text = "Este es un texto con una entidad y otra entidad."
    span_texts = ["entidad", "otra entidad"]
    orig_spans = [
        Span(text="entity", start=23, end=24, label="ENTITY"),
        Span(text="entity", start=37, end=42, label="ENTITY"),
    ]

    example = match_example(lang, text, span_texts, orig_spans)

    assert example == Example(
        text="Este es un texto con una entidad y otra entidad.",
        spans=[
            Span(text="entidad", start=25, end=32, label="ENTITY", token_start=6, token_end=7),
            Span(
                text="otra entidad", start=35, end=47, label="ENTITY", token_start=8, token_end=10
            ),
        ],
        tokens=[
            Token(text="Este", start=0, end=4, id=0),
            Token(text="es", start=5, end=7, id=1),
            Token(text="un", start=8, end=10, id=2),
            Token(text="texto", start=11, end=16, id=3),
            Token(text="con", start=17, end=20, id=4),
            Token(text="una", start=21, end=24, id=5),
            Token(text="entidad", start=25, end=32, id=6),
            Token(text="y", start=33, end=34, id=7),
            Token(text="otra", start=35, end=39, id=8),
            Token(text="entidad", start=40, end=47, id=9),
            Token(text=".", start=47, end=48, id=10),
        ]
    )
