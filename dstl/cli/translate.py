from pathlib import Path

import srsly
from wasabi import msg

from ..translate import TransformersMarianTranslator, translate_ner_batch
from ..types import Example, Task


def translate(
    input_path: Path,
    model_name_or_path: str,
    source_lang: str,
    target_lang: str,
    output_path: Path = None,
    force: bool = False,
    task: Task = Task.NER,
) -> None:
    """Translate dataset

    Args:
        input_path (Path): Path to file JSONL file with annotated data
        model_name_or_path (str): Model name or path of MarianMT based model using HuggingFace Transformers
        source_lang (str): Source language of text.
        target_lang (str): Target language of text.
        output_path (Path): Output path to save data to.
        force (bool): Force output overwrite and creation.
        task (Task): NLP Task format of the data. 
            e.g. "NER", "Classification". Currently, only "NER" is supported
    """

    if input_path.suffix != ".jsonl":
        raise ValueError("Only accepting JSONL data in the Prodigy Annotation format.")

    if output_path and not output_path.exists():
        if output_path.is_file() and not force:
            raise ValueError(
                "Output path already exists. To overwrite add the flag --force to your command"
            )

        output_path.parent.mkdir(exist_ok=True, parents=True)

    msg.text("Loading input examples")

    raw_examples = srsly.read_jsonl(input_path)
    examples = [Example(**e) for e in raw_examples]

    msg.good(f"Successfully loaded {len(examples)}")

    msg.text(f"Translating examples.")

    translator = TransformersMarianTranslator(
        model_name_or_path, source_lang=source_lang, target_lang=target_lang
    )
    examples_t = translate_ner_batch(examples, translator.pipe, target_lang)

    srsly.write_jsonl(output_path, (e.dict() for e in examples_t))
