from pathlib import Path

import pytest

from pad_synth_core.ontology import (
    Ontology,
    OntologyLintError,
    load_ontology,
)


GOOD_YAML = """
version: "2026-05-11"
attack_type: print
axes:
  paper_type:
    type: categorical
    values: [matte, glossy, photo]
    weights: [0.5, 0.4, 0.1]
    provenance:
      paper: "Galbally 2014"
      doi: "10.0/example"
  print_dpi:
    type: categorical
    values: [150, 300, 600, 1200]
    weights: [0.1, 0.4, 0.4, 0.1]
    provenance:
      paper: "Example Vendor Spec 2023"
      url: "https://example.com/spec"
  tilt_degrees:
    type: uniform
    low: -30.0
    high: 30.0
    provenance:
      paper: "Boulkenafet 2017 OULU-NPU paper"
      doi: "10.0/oulu"
"""


BAD_YAML_NO_PROVENANCE = """
version: "2026-05-11"
attack_type: print
axes:
  paper_type:
    type: categorical
    values: [matte, glossy]
    weights: [0.5, 0.5]
"""


def test_load_ontology_parses_axes(tmp_path: Path):
    p = tmp_path / "print.yaml"
    p.write_text(GOOD_YAML)
    ont = load_ontology(p)
    assert ont.attack_type == "print"
    assert "paper_type" in ont.axes
    assert ont.version == "2026-05-11"


def test_lint_rejects_axis_without_provenance(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(BAD_YAML_NO_PROVENANCE)
    with pytest.raises(OntologyLintError) as exc:
        load_ontology(p)
    assert "paper_type" in str(exc.value)
    assert "provenance" in str(exc.value)


def test_sample_categorical_is_deterministic(tmp_path: Path):
    p = tmp_path / "print.yaml"
    p.write_text(GOOD_YAML)
    ont = load_ontology(p)
    import numpy as np

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    params1 = ont.sample_params(rng1)
    params2 = ont.sample_params(rng2)
    assert params1 == params2
    assert params1["paper_type"] in {"matte", "glossy", "photo"}
    assert -30.0 <= params1["tilt_degrees"] <= 30.0
