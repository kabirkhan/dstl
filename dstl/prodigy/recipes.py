from typing import List

import prodigy
from prodigy.core import connect
from prodigy.util import set_hashes, split_string
from wasabi import msg

from ..translate import TransformersMarianTranslator, translate_ner_batch
from ..types import Example


@prodigy.recipe(
    "ner.translate",
    in_sets=("Comma-separated datasets to merge", "positional", None, split_string),
    out_set=("Name of new dataset for merged data", "positional", None, str),
    model_name_or_path=(
        "Model name or path of MarianMT based model using HuggingFace Transformers",
        "positional",
        None,
        str,
    ),
    source_lang=("Source language for translation", "option", "sl", str),
    target_lang=("Target language for translation", "option", "tl", str),
    dry=("Perform a dry run", "flag", "D", bool),
)
def ner_translate(
    in_sets: List[str],
    out_set: str,
    model_name_or_path: str,
    source_lang: str,
    target_lang: str,
    dry: bool = False,
) -> None:
    translator = TransformersMarianTranslator(
        model_name_or_path, source_lang=source_lang, target_lang=target_lang
    )

    DB = connect()
    for set_id in in_sets:
        if set_id not in DB:
            msg.fail(f"Can't find dataset '{set_id}' in database", exits=1)
    if out_set in DB and len(DB.get_dataset(out_set)):
        msg.fail(
            f"Output dataset '{out_set}' already exists and includes examples",
            f"This can lead to unexpected results. Please use a new dataset.",
            exits=1,
        )
    if out_set not in DB:
        if not dry:
            DB.add_dataset(out_set)
        msg.good(f"Created dataset '{out_set}'")

    matched_examples_t = []
    mismatched_examples_t = []

    for set_id in in_sets:
        msg.text(f"RECIPE: Translating and merging examples from '{set_id}'")
        raw_examples = DB.get_dataset(set_id)
        examples = [Example(**e) for e in raw_examples]
        examples_t = translate_ner_batch(
            examples, translate_f=translator.pipe, target_lang=target_lang
        )
        for e, e_t in zip(examples, examples_t):
            if len(e.spans) != len(e_t.spans):
                mismatched_examples_t.append(e_t)
            else:
                matched_examples_t.append(e_t)

        msg.text(f"RECIPE: Translated {len(matched_examples_t)} examples from '{set_id}'")
        msg.text(
            f"RECIPE: Found {len(mismatched_examples_t)} examples with mismatched spans after translation from '{set_id}'"
        )

    matched_examples_t = set_hashes(matched_examples_t)

    dry = False
    if not dry:
        DB.add_examples(matched_examples_t, datasets=[out_set])
    msg.good(
        f"Translated and merged {len(matched_examples_t)} examples from {len(in_sets)} datasets",
        f"Created translated and merged dataset '{out_set}'",
    )
