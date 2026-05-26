from pad_synth_face.pipeline import _canonical_ontology_version


class _FakeOnt:
    def __init__(self, version):
        self.version = version


class _FakeModule:
    def __init__(self, version):
        self.ontology = _FakeOnt(version)


def test_prefers_print_when_present():
    mods = {"print": _FakeModule("P"), "replay": _FakeModule("R")}
    assert _canonical_ontology_version(mods) == "P"


def test_falls_back_to_priority_then_alpha():
    # No print: priority order is print -> replay -> mask.
    mods = {"mask": _FakeModule("M"), "replay": _FakeModule("R")}
    assert _canonical_ontology_version(mods) == "R"


def test_mask_only_does_not_raise():
    mods = {"mask": _FakeModule("M")}
    assert _canonical_ontology_version(mods) == "M"


def test_unknown_attack_uses_alphabetical_first():
    mods = {"zeta": _FakeModule("Z"), "alpha": _FakeModule("A")}
    assert _canonical_ontology_version(mods) == "A"
