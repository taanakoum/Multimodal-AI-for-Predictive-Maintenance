"""Context sentences and fault descriptions per dataset (plan, "Rules for the text inputs").

Each dataset has one YAML file here:

* ``context``: a template describing the machine, sensors, speed and load. It is filled from
  a recording's ``condition`` and must never name or hint at the state.
* ``templates`` and ``classes``: 3 paraphrases per fault type, expanded per class key
  (``label`` or ``label/severity``). They describe the fault and its size and never mention
  speed or load, so no class can be matched through the context text.

The Qwen3 class prototypes are the L2-normalized mean of a class's paraphrase embeddings
(:mod:`pdmfm.features`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

TEXT_DIR = Path(__file__).resolve().parent

# Words that would put the state into the context, and operating conditions that would put
# the context into the class descriptions. Checked by tests/test_texts.py.
STATE_WORDS = re.compile(
    r"\b(defect\w*|fault\w*|damag\w*|crack\w*|pitting|wear|worn|dull\w*|sharp|healthy|normal\w*|"
    r"unbalanc\w*|imbalanc\w*|misalign\w*|race|failure|short.circuit\w*|stuck|locked|broken)\b",
    re.IGNORECASE,
)
CONDITION_WORDS = re.compile(r"\b(rpm|nm|hp|kn|torque|load\w*|speed|hz)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetTexts:
    dataset: str
    label_space: str | None
    context_template: str
    descriptions: dict[str, tuple[str, ...]]  # class key -> paraphrases

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.descriptions)

    def context(self, condition: dict) -> str:
        """The context sentence for one recording's operating condition."""
        try:
            return self.context_template.format_map(condition)
        except KeyError as error:
            raise KeyError(f"{self.dataset}: context needs condition field {error}") from None


def class_key(label: str, severity: str | None = None) -> str:
    """Class key in a label space: ``label`` or ``label/severity``."""
    return label if not severity else f"{label}/{severity}"


@cache
def load(dataset: str) -> DatasetTexts:
    path = TEXT_DIR / f"{dataset}.yaml"
    if not path.is_file():
        raise KeyError(f"no texts for dataset {dataset!r}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    severity_text = spec.get("severity_text", {})
    templates = spec.get("templates") or {}
    descriptions = {}
    for key in spec.get("classes") or []:
        label, _, severity = key.partition("/")
        if label not in templates:
            raise ValueError(f"{path.name}: class {key!r} has no template {label!r}")
        shown = severity_text.get(severity, severity)
        descriptions[key] = tuple(t.format(severity=shown) for t in templates[label])
    return DatasetTexts(
        dataset=spec["dataset"],
        label_space=spec.get("label_space"),
        context_template=" ".join(spec["context"].split()),
        descriptions=descriptions,
    )


def datasets() -> list[str]:
    return sorted(p.stem for p in TEXT_DIR.glob("*.yaml"))
