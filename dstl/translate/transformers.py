from typing import Iterable, List, Optional

from spacy.util import minibatch
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

from .base import BaseTranslator


class TransformersMarianTranslator(BaseTranslator):
    """TransformersMarianTranslator uses the MarianMTModel from the transformers project
    to translate text to/from any supported model in Marian MT."""

    name = "transformers"

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
        super().__init__(source_lang, target_lang)

    def _predict(self, texts: List[str], batch_size: Optional[int] = 8) -> Iterable[str]:
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
