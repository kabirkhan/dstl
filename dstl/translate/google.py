from typing import Iterable, List, Optional

import httpx
from spacy.util import minibatch
from tqdm.auto import tqdm

from .base import BaseTranslator


class GoogleTranslator(BaseTranslator):
    """GoogleTranslator uses the Google Cloud Translation API to translate documents
    from source_lang to target_lang."""

    name = "google"

    def __init__(
        self,
        api_key: str,
        source_lang: str,
        target_lang: str,
        translate_url: str = "https://translation.googleapis.com/language/translate/v2",
    ):
        """Initialize an instance of GoogleTranslator

        Args:
            api_key (str): API Key for your instance of the Translate API
            source_lang (str, optional): Source language to translate from
            target_lang (str, optional): Language to translate to
            translate_url (str): URL of translator endpoint
        """
        self._translate_url = translate_url
        self._default_params = {"key": api_key}

        super().__init__(source_lang, target_lang)

    def _predict(self, texts: List[str], batch_size: Optional[int] = 1000) -> Iterable[str]:

        translated_texts = []
        with tqdm(total=len(texts)) as pbar:
            for batch in minibatch(texts, batch_size):
                json_body = [
                    {
                        "q": text,
                        "source": self.source_lang,
                        "target": self.target_lang,
                        "format": "text",
                    }
                    for text in batch
                ]
                res = httpx.post(self._translate_url, params=self._default_params, json=json_body)
                data = res.json()
                for doc in data["translations"]:
                    translated_texts.append(doc["text"])
                    pbar.update(1)

        return translated_texts
