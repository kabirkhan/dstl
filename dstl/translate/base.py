from abc import ABC, abstractmethod
from typing import Iterable, List, Optional


class BaseTranslator(ABC):
    """Base Translator interface."""

    def __init__(self, source_lang: str, target_lang: str):
        """Initialize an instance of BaseTranslator

        Args:
            source_lang (str, optional): Source language to translate from
            target_lang (str, optional): Language to translate to
        """
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __call__(self, text: str) -> str:
        """Translate a single text document

        Args:
            text (str): Text to translate in source language

        Returns:
            str: Translated text in target language
        """
        return list(self.pipe([text]))[0]

    def pipe(self, texts: List[str], batch_size: Optional[int] = 8) -> Iterable[str]:
        return self._predict(texts, batch_size)

    @abstractmethod
    def _predict(self, texts: List[str], batch_size: Optional[int] = 8) -> Iterable[str]:
        """Translate a batch of text documents

        Args:
            texts (Iterable[str]): Texts to translate in source language
            batch_size (int): Batch size for feeding texts to model

        Returns:
            Iterable[str]: Translated texts in target language
        """
        raise NotImplementedError
