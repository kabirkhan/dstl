from typing import Iterable, List, Optional

import httpx
from spacy.util import minibatch
from tqdm.auto import tqdm

from .base import BaseTranslator


class AzureTranslator(BaseTranslator):
    """AzureTranslator uses the Microsoft Azure Translation API to translate documents
    from source_lang to target_lang."""

    name = "azure"

    def __init__(
        self,
        api_key: str,
        source_lang: str,
        target_lang: str,
        translate_url: str = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0",
    ):
        """Initialize an instance of AzureTranslator

        Args:
            api_key (str): API Key for your instance of the Translate API
            source_lang (str, optional): Source language to translate from
            target_lang (str, optional): Language to translate to
            translate_url (str): URL of translator endpoint
        """
        self._translate_url = translate_url
        self._default_params = {"from": source_lang, "to": target_lang}
        self._default_headers = {"Ocp-Apim-Subscription-Key": api_key}

        super().__init__(source_lang, target_lang)

    def _predict(self, texts: List[str], batch_size: Optional[int] = 1000) -> Iterable[str]:

        translated_texts = []
        with tqdm(total=len(texts)) as pbar:
            for batch in minibatch(texts, batch_size):
                json_body = [{"text": text} for text in batch]
                res = httpx.post(
                    self._translate_url,
                    params=self._default_params,
                    headers=self._default_headers,
                    json=json_body,
                )
                data = res.json()
                for doc in data:
                    translation = doc["translations"][0]
                    translated_texts.append(translation["text"])
                    pbar.update(1)

        return translated_texts
