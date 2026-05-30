% PAD B2 — Pretrained-Backbone Capacity Slot Design
% Add ImageNet-pretrained ResNet18 as a new L4 model in `FACTORIES`. The 2026-05-29 capacity spike showed pretrained ResNet18 cuts mask-only cross-domain EER from 0.291 (from-scratch) to **0.045** at L3·D3·224 — better than the 64×64 baseline. Roll the win out as a focused 18-cell sweep additive to the existing 224 report.
% 2026-05-30

---

## 1. Purpose and audience

The 2026-05-29 A1 resolution-bump sweep was a negative result on its own: cross-domain EER got worse at 224×224 because the existing `FACTORIES` models (TinyCNN / SmallCNN / ResNet18 from scratch) are under-capacity for 12× the pixel count and can no longer fit the synthetic training set (L3·D3 in-domain EER jumped 0.000 → 0.465). A same-day capacity spike on the same cell tested three variants:

| Model | params | in-domain EER | cross-domain EER | ACER@5% APCER |
|---|---|---|---|---|
| TinyCNN (baseline) | 1.4k | 0.461 | 0.272 | 0.348 |
| TinyCNNWide (4× wider) | 19.5k | 0.363 | 0.310 | 0.353 |
| ResNet18 from scratch | 11M | 0.416 | 0.287 | 0.322 |
| **ResNet18 ImageNet-pretrained** | 11M | **0.099** | **0.045** | **0.052** |

The pretrained variant is the only configuration in the project's history with a usable operating point at the ISO 30107-3 5 % APCER target. It also beats the 64×64 L3·D3 mask-only baseline of 0.089 by ~2×. The amount of capacity matters less than the *form* — ImageNet pretrained features carry inductive bias that from-scratch weights, at any width, can't match on 4k synthetic samples.

This spec rolls that one-cell finding out as a focused 18-cell sweep: add `make_resnet18_pretrained` to `FACTORIES["L4"]`, run the 9 mask + 9 mix L4 cells on the Spark, append an L4 column to the existing 224 report. L1/L2/L3 stay unchanged.

Audience: future maintainers; the next sub-projects on the queue (A2 capture-realism — now incremental on top of B2; DFDC sweep; Tier-B real-attack benchmark).

## 2. The question this answers

| Behaviour on the 18-cell L4 sweep | What it tells us |
|---|---|
| L4·D3 cross-domain EER stays near the spike's 0.045 across both mask and mix sweeps | The spike generalises. B2 is a real win and the project has a first deployable PAD detector. Roll forward; A2 becomes the incremental capture-realism layer. |
| L4·D3 lands 0.10–0.20 (worse than spike but better than L3 from-scratch) | The spike was on the favourable seed. B2 is still a clear win but with more variance than expected — investigate (LR schedule, normalization, longer training). |
| L4·D3 cross-domain ~ L3 from-scratch (~0.27) | Spike was a fluke. B2 is not the lever; reconsider A2 alone, longer training, or other backbones. Less likely given the spike's magnitude. |
| Any cell collapses to 0.000 cross-domain | New artifact emerged at L4 (e.g., pretrained features lock onto a synthetic generator fingerprint the smaller models couldn't exploit). Apply the v2/v2.1 anti-fingerprint playbook. Unlikely given the 0.045 spike result is non-degenerate (APCER 0.019, BPCER 0.084 — neither pinned at 0). |

Decision rule (mirrors all prior sweeps): `no cross-domain cell mean ≤ 0.001` = artifact-free.

## 3. Architecture and files

The smallest possible delivery: one new factory function, one entry added to `FACTORIES`, tests, and the sweep run. No changes to the eval baseline, the sweep script, the determinism golden, or the configs.

| Change | File | Note |
|---|---|---|
| New factory | `pad-synth-core/src/pad_synth_core/eval/models_zoo.py` | `make_resnet18_pretrained()` using `ResNet18_Weights.IMAGENET1K_V1`; final fc replaced with `Linear(512, 2)`. Added to `FACTORIES` as `"L4"`. |
| Factory unit tests | `pad-synth-core/tests/test_models_zoo.py` | `FACTORIES["L4"]` exists; factory returns nn.Module; forward pass on `(1, 3, 224, 224)` returns shape `(1, 2)` |
| Pretrained smoke test | `pad-synth-core/tests/test_baseline_extensions.py` | One-cell `train_and_cross_domain_eval(..., model_factory=make_resnet18_pretrained, epochs=1, device="cpu")` on the existing fixture-built dataset; assert finite EER. Skipped if pretrained weights can't be downloaded (network-dependent). |
| (Optional) extend default-grid CAPACITIES | `scripts/spark_sweep.py` | Append `"L4"` to `CAPACITIES = ("L1","L2","L3")` so future full-grid sweeps include L4. Not strictly required since the `--cells` filter handles this run; one-line change, documented in §6. |

**No changes** to: `eval/baseline.py`, `eval/metrics.py`, training loop, dataset loaders, ISO metrics, configs, determinism golden, or the report's prior sections.

## 4. Model factory

```python
from torchvision.models import resnet18, ResNet18_Weights


def make_resnet18_pretrained() -> nn.Module:
    """ResNet18 with ImageNet-pretrained weights; final fc replaced with Linear(512, 2).
    Same head-swap pattern as the existing make_resnet18; weights flag is the only delta."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


FACTORIES = {
    "L1": make_tiny_cnn,
    "L2": make_small_cnn,
    "L3": make_resnet18,
    "L4": make_resnet18_pretrained,
}
```

The first call downloads `resnet18-f37072fd.pth` (~45 MB) into `~/.cache/torch/hub/checkpoints/`. Subsequent calls are cached. The Spark already has it cached from the spike.

## 5. Training configuration (unchanged from spike)

Match the configuration that produced the dramatic spike result, exactly:
- **Optimiser:** Adam, lr=1e-3 (the existing `train_and_cross_domain_eval` default).
- **Epochs:** 8 (sweep default — same as the spike).
- **Batch size:** 32 (sweep default).
- **Input normalisation:** **None** — divide by 255 only, matching the existing eval pipeline. The spike worked dramatically without ImageNet mean/std normalisation; adding it is a possible future optimisation but out of scope for this cycle (would change the pipeline for all models).
- **Subject-disjoint dev split** (from the eval-metrics upgrade) — applies as today.
- **Threshold fixed on dev at APCER ≤ 5 %** (from the eval-metrics upgrade) → applied to cross-domain test.

## 6. Sweep invocation

Two 9-cell sweeps via the existing `spark_sweep.py` with the `--cells` filter:

```bash
ssh swells@spark-50d2.local '
cd ~/ml/projects/pad-spark
CELLS=$(python3 -c "print(\",\".join(f\"L4:{D}:{s}\" for D in (\"D1\",\"D2\",\"D3\") for s in (0,1,2)))")
# Mask-only L4 sweep
.venv/bin/python scripts/spark_sweep.py \
  --set-a-d1 datasets/mask_seta_d1 --set-b-d1 datasets/mask_setb_d1 \
  --set-a-d2 datasets/mask_seta_d2 --set-b-d2 datasets/mask_setb_d2 \
  --set-a-d3 datasets/mask_seta_d3 --set-b-d3 datasets/mask_setb_d3 \
  --set-a-d4 datasets/mask_seta_d3 --set-b-d4 datasets/mask_setb_d3 \
  --output-dir docs/superpowers/reports/2026-05-22-pad-spark-sweep-results/runs_mask_224_L4 \
  --cells "$CELLS" --device cuda
# Integrated L4 sweep (same command, mix_ prefixes + runs_mix_224_L4 output)
'
```

Per the spike, one D3 cell on GB10 takes ~60 s; full 18 cells projected at ~10–15 min wall-time.

If `scripts/spark_sweep.py:CAPACITIES` is extended to `("L1","L2","L3","L4")`, the default no-`--cells` invocation would sweep 36 cells (L1–L4 × D1–D3 × 3 seeds). Optional — this cycle uses the `--cells` filter and doesn't depend on the tuple change.

## 7. Report

Append a new section to `docs/superpowers/reports/2026-05-22-pad-spark-sweep-results.md`:

```markdown
## 2026-05-XX update — B2 (pretrained ResNet18 backbone) result

[Setup line: spike → full sweep, code SHA, sweep wall-time]

### L4 = pretrained ResNet18 cross-domain EER (mean ± std across 3 seeds)

|  | mask-only | integrated (print+replay+mask) |
|---|---|---|
| L4·D1 | … | … |
| L4·D2 | … | … |
| L4·D3 | … | … |

[in-domain table same shape]
[ACER@5%APCER table same shape — first column of usable operating points]

### Headline finding
[Either: spike generalised → first deployable PAD config; or: spike-vs-sweep delta with diagnosis]

### Comparison
| Cell | L3 from-scratch | L4 pretrained | Δ |
|---|---|---|---|
| mask·D3 | 0.291 | … | … |
| mix·D3  | 0.225 | … | … |

### Phase recommendation update
[Where this leaves the lever queue]
```

Old report sections (2026-05-22 v1/v2/v2.1, 2026-05-27 real-bonafide & mask, 2026-05-29 A1) stay **immutable** as the from-scratch baseline.

## 8. Testing

`pad-synth-core/tests/test_models_zoo.py` gains:

```python
def test_l4_factory_in_registry_and_callable():
    from pad_synth_core.eval.models_zoo import FACTORIES
    assert "L4" in FACTORIES
    m = FACTORIES["L4"]()
    assert isinstance(m, nn.Module)


def test_l4_factory_forward_returns_2logits():
    from pad_synth_core.eval.models_zoo import make_resnet18_pretrained
    m = make_resnet18_pretrained()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)
```

`pad-synth-core/tests/test_baseline_extensions.py` gains a pretrained-smoke test that skips if the weights download fails (network-gated).

The L4 factory unit tests will trigger the ~45 MB pretrained weights download once per CI environment — acceptable given B2 is the lever and the cached weights persist.

## 9. Compute and rollout

- **Sweep wall-time:** ~10–15 min (18 cells, ResNet18 ~60 s per D3 cell on GB10; D1/D2 faster).
- **Disk:** no new datasets — reuses the 12 `mask_*`/`mix_*` 224×224 datasets from the A1 cycle. Only per-cell JSON + summary CSV in `runs_mask_224_L4/` and `runs_mix_224_L4/` (committed).
- **Network:** one-time ResNet18 weights download (~45 MB) on the Spark, already cached from the spike.

## 10. Out of scope

- **ImageNet input normalisation** — eval pipeline currently divides by 255 only; the spike worked dramatically without ImageNet mean/std. Adding it would change inputs for L1/L2/L3 too. Possible future incremental optimisation.
- **Per-model LR / training schedule** — Adam lr=1e-3 worked in the spike. Per-model tuning is YAGNI for the first B2 deliverable.
- **ResNet50 / other backbones** — B2-extended; queued for later if L4 saturates.
- **A2 (sensor capture-realism)** — incremental on top of B2; the next sub-project after this cycle.
- **B1 (synth-pretrain → real-finetune curve)** — its value at 224 depends on B2 being a strong initialiser. Run after B2 ships.
- **Re-run of L1/L2/L3 cells** — those numbers are immutable in the prior report section. No need to re-run.
