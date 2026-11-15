"""Context sentences and fault descriptions (pdmfm/texts/*.yaml)."""

from __future__ import annotations

import pytest

from pdmfm import texts
from pdmfm.datasets.kaist import parse_condition
from pdmfm.datasets.najafi import parse_folder
from pdmfm.datasets.nonastreda import CLASS_NAMES
from pdmfm.datasets.paderborn import DAMAGE
from pdmfm.sources import CWRU_FILES

KAIST_STEMS = [
    "0Nm_Normal", "0Nm_BPFI_03", "0Nm_BPFI_10", "0Nm_BPFI_30", "0Nm_BPFO_03", "0Nm_BPFO_10",
    "0Nm_BPFO_30", "0Nm_Misalign_01", "0Nm_Misalign_03", "0Nm_Misalign_05",
    "0Nm_Unbalance_0583mg", "0Nm_Unbalance_1169mg", "0Nm_Unbalance_1751mg",
    "0Nm_Unbalance_2239mg", "0Nm_Unbalance_3318mg",
]  # fmt: skip
CONDITIONS = {
    "cwru": {"rpm": 1797.0, "load_hp": 0},
    "kaist": {"rpm": 3010.0, "load_nm": 4},
    "paderborn": {"rpm": 1500.0, "torque_nm": 0.7, "radial_n": 1000.0},
    "xjtu_sy": {"rpm": 2100.0, "radial_kn": 12.0},
    "phm2010": {"rpm": 10400.0},
    "nonastreda": {},
    "najafi_thermal": {},
    "matwi": {"material": "C45 steel", "inserts": 1, "speed_text": "2547 rpm"},
}


def test_every_dataset_has_texts():
    assert set(texts.datasets()) == set(CONDITIONS)


@pytest.mark.parametrize("dataset", sorted(CONDITIONS))
def test_context_names_conditions_never_the_state(dataset):
    sentence = texts.load(dataset).context(CONDITIONS[dataset])
    assert "{" not in sentence and len(sentence) > 40
    assert texts.STATE_WORDS.search(sentence) is None, texts.STATE_WORDS.search(sentence)


@pytest.mark.parametrize("dataset", sorted(CONDITIONS))
def test_descriptions_name_the_state_never_conditions(dataset):
    spec = texts.load(dataset)
    for key, paraphrases in spec.descriptions.items():
        assert len(paraphrases) >= 3, key
        assert len(set(paraphrases)) == len(paraphrases), f"{key}: repeated paraphrase"
        for sentence in paraphrases:
            assert "{" not in sentence
            assert texts.CONDITION_WORDS.search(sentence) is None, (key, sentence)


def test_descriptions_differ_between_classes():
    for dataset in texts.datasets():
        spec = texts.load(dataset)
        all_sentences = [s for p in spec.descriptions.values() for s in p]
        assert len(set(all_sentences)) == len(all_sentences), dataset


def test_reader_labels_map_onto_the_label_spaces():
    cwru = texts.load("cwru")
    for kind, size, _ in CWRU_FILES.values():
        severity = None if kind == "normal" else f"{size:.3f} in"
        assert texts.class_key(kind, severity) in cwru.descriptions
    kaist = texts.load("kaist")
    keys = {
        texts.class_key(**{k: v for k, v in parse_condition(s).items() if k != "load_nm"})
        for s in KAIST_STEMS
    }
    assert keys == set(kaist.descriptions)
    paderborn = texts.load("paderborn")
    labels = {label for label, _, _ in DAMAGE.values()}
    assert labels - set(paderborn.descriptions) == {"combined"}  # KB27: outside the space
    assert set(CLASS_NAMES) == set(texts.load("nonastreda").descriptions)
    folders = ("Noload", "A10", "A&C&B30", "Fan", "Rotor-0")
    assert {parse_folder(f)[0] for f in folders} == set(texts.load("najafi_thermal").descriptions)


def test_severity_text_spells_out_cwru_sizes_in_millimetres():
    sentence = texts.load("cwru").descriptions["outer_race/0.014 in"][0]
    assert "0.36 mm (0.014 in)" in sentence


def test_missing_condition_field_is_reported():
    with pytest.raises(KeyError, match="load_nm"):
        texts.load("kaist").context({"rpm": 3010.0})


def test_unknown_dataset():
    with pytest.raises(KeyError):
        texts.load("no_such_dataset")


def test_regression_datasets_have_no_classes():
    assert texts.load("xjtu_sy").classes == () and texts.load("phm2010").label_space is None


def test_matwi_shares_the_milling_wear_classes():
    """MATWI uses Nonastreda's classes and descriptions, so both milling sets share one
    text-anchored label space (their prototypes are identical)."""
    from pdmfm.datasets.matwi import CLASS_NAMES as MATWI_CLASSES

    matwi, nonastreda = texts.load("matwi"), texts.load("nonastreda")
    assert set(MATWI_CLASSES) == set(matwi.descriptions)
    assert matwi.descriptions == nonastreda.descriptions
