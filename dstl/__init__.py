"""DataSet TransLation (DSTL) provides utilities to translate annotated natural language data from one language to another."""

__version__ = "0.0.2"

from .translate import TransformersMarianTranslator
from .translate.core import translate_ner_batch

try:
    # This needs to be imported in order for the entry points to be loaded
    from .prodigy import recipes as prodigy_recipes  # noqa: F401
except ImportError as e:
    pass
