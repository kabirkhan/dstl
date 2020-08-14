from typing import Callable, Iterable, List, Optional

import spacy
from spacy.util import minibatch
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

from .types import Example, Span, Token


class TransformersMarianTranslator:
    """TransformersMarianTranslator uses the MarianMTModel from the transformers project
    to translate text to/from any supported model in Marian MT."""

    def __init__(self, model_name_or_path: str, source_lang: str, target_lang: str):
        """Initialize an instance of TransformersMarianTranslator

        Args:
            model_name_or_path (str): Pretrained model name or path of Marian MT based model
                e.g. "Helsinki-NLP/opus-mt-en-ROMANCE"
            source_lang (str, optional): Source language to translate from
            target_lang (str, optional): Language to translate to

        Raises:
            ValueError: Target language is ambiguous given the model and no target_lang 
                is directly specified.
        """
        self.tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)
        self.model = MarianMTModel.from_pretrained(model_name_or_path)
        self.source_lang = source_lang
        self.target_lang = target_lang

        # source_langs = self.tokenizer.init_kwargs["source_lang"].split("+")
        # if len(source_langs) == 1:
        #     self.source_lang = source_langs[0]

        # target_langs = self.tokenizer.init_kwargs["target_lang"].split("+")
        # if len(target_langs) > 1 and not self.target_lang:
        #     raise ValueError("Please provide a target language.")
        # elif len(target_langs) == 1:
        #     self.target_lang = target_langs[0]

    def __call__(self, text: str) -> str:
        """Translate a single text document

        Args:
            text (str): Text to translate in source language

        Returns:
            str: Translated text in target language
        """
        return list(self.pipe([text]))[0]

    def pipe(self, texts: List[str], batch_size: Optional[int] = 8) -> Iterable[str]:
        """Translate a batch of text documents

        Args:
            texts (Iterable[str]): Texts to translate in source language
            batch_size (int): Batch size for feeding texts to model

        Returns:
            Iterable[str]: Translated texts in target language
        """
        prefix = f">>{self.target_lang}<< "
        texts = [prefix + text for text in texts]
        with tqdm(total=len(texts)) as pbar:
            for batch in minibatch(texts, batch_size):
                encoded_inputs = self.tokenizer.prepare_translation_batch(batch)
                translated = self.model.generate(**encoded_inputs)
                tgt_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                yield from tgt_texts
                pbar.update(batch_size)


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

    with tqdm(total=len(examples)) as pbar:
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
            pbar.update(1)
