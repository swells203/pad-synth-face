# Determinism Golden Set

`tests/test_determinism_golden.py` regenerates a fixed 16-sample run from a
pinned config and checks every output's SHA-256 against `golden_hashes.json`.

If a code change is *intentionally* expected to change outputs, regenerate the
golden file:

```bash
PAD_SYNTH_UPDATE_GOLDEN=1 python -m pytest tests/test_determinism_golden.py
```

Then commit `golden_hashes.json` together with the code change so reviewers see
the diff.
