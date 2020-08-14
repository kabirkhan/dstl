from typing import Callable, Iterable, List, Optional

import spacy
from tqdm.auto import tqdm

from ..types import Example, Span, Token


def match_example(
    lang: str, text: str, span_texts: List[str], spans: List[Span], case_sensitive: bool = True
) -> Example:
    """Match Example with provided spans using spaCy EntityRuler

    Args:
        lang (str): Target spaCy language 
        text (str): Example to text to match
        span_texts (List[str]): Span text to identify in text
        spans (List[Span]): Original spans in source language
        case_sensitive (bool, optional): Consider case during matching.

    Returns:
        Example: Tokenized Example in target language with spans set correctly
    """
    nlp = spacy.blank(lang)
    ruler = nlp.create_pipe(
        "entity_ruler", {"phrase_matcher_attr": "ORTH" if case_sensitive else "LOWER"}
    )
    patterns = [{"label": s.label, "pattern": st_text} for s, st_text in zip(spans, span_texts)]

    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)

    doc = nlp(text)

    return Example(
        text=doc.text,
        spans=[
            Span(
                text=e.text,
                start=e.start_char,
                end=e.end_char,
                label=e.label_,
                token_start=e.start,
                token_end=e.end,
            )
            for e in doc.ents
        ],
        tokens=[Token(text=t.text, start=t.idx, end=t.idx + len(t), id=t.i) for t in doc],
    )


def translate_ner_batch(
    examples: Iterable[Example],
    translate_f: Callable[[List[str], Optional[int]], Iterable[str]],
    target_lang: str,
    case_sensitive: bool = True,
    batch_size: int = 8,
    show_progress: bool = True,
) -> Iterable[Example]:
    """Translate a batch of labeled Named Entity Recognition (NER) examples in the into `target_lang`

    Args:
        examples (Iterable[Example]): Input examples
        translate_f (Callable[[Iterable[str]], Iterable[str]]): 
            Translation function that operates on batch of text
        target_lang (str): Target language code without locale available in spaCy. 
            See: for full list
        case_sensitive (bool, optional): Use case sensitive matching for translation of 
            spans matches in translated examples.
        batch_size (int): Batch size for iterating through examples
        show_progress (bool, optional): Show tqdm progress bar

    Returns:
        Iterable[Example]: Examples translated and tokenized in `target_lang`
    """
    examples = list(examples)

    offsets = [0]
    texts_to_translate = []

    for example in examples:
        example_texts = [example.text] + [s.text for s in example.spans]
        texts_to_translate += example_texts
        offsets.append(offsets[-1] + len(example_texts))

    translated_texts = list(translate_f(texts_to_translate, batch_size))

    for i in tqdm(range(1, len(offsets))):
        orig_example = examples[i - 1]
        e_texts_t = translated_texts[offsets[i - 1] : offsets[i]]
        example_text_t = e_texts_t[0]
        span_texts_t = e_texts_t[1:]
        example_t = match_example(
            target_lang, example_text_t, span_texts_t, orig_example.spans, case_sensitive
        )

        yield example_t
