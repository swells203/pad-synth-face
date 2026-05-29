"""Deterministic procedural-bonafide fixture (extracted from conftest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def build_fixture_bonafide(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for identity in range(8):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir(exist_ok=True)
        for sample in range(2):
            base = rng.integers(50, 200, size=3)
            arr = np.tile(base, (64, 64, 1)).astype(np.uint8)
            noise = rng.integers(-20, 20, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root


# Fitzpatrick-inspired skin-tone base colors (RGB). Not personally identifying;
# these are abstract palette anchors approximating ranges documented in
# Krishnapriya et al., "Issues Related to Face Recognition Accuracy Varying
# Based on Race and Skin Tone", IEEE Trans. Tech. Soc. 2020.
_SKIN_TONE_PALETTE: list[tuple[int, int, int]] = [
    (244, 219, 196),  # very light
    (224, 192, 165),
    (199, 158, 125),
    (170, 124, 92),
    (133, 90, 60),
    (95, 60, 38),
    (215, 175, 135),  # warm light
    (185, 140, 105),
    (155, 110, 80),
    (120, 85, 60),
    (235, 200, 170),
    (205, 165, 130),
    (175, 130, 95),
    (145, 105, 75),
    (115, 85, 65),
    (90, 65, 45),
]


def _oval_mask(h: int, w: int) -> np.ndarray:
    """Face-shaped Gaussian falloff: 1.0 at center, ~0.3 at the corners."""
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Wider horizontally is less face-shaped; use slight ovalness.
    ry, rx = h * 0.42, w * 0.36
    r = np.sqrt(((yv - cy) / ry) ** 2 + ((xv - cx) / rx) ** 2)
    mask = np.exp(-(r**2) * 1.4)
    return np.clip(mask, 0.3, 1.0)


def _eye_region_darken(h: int, w: int) -> np.ndarray:
    """Darken patches at expected eye y-band (~30-45% from top)."""
    out = np.ones((h, w), dtype=np.float32)
    y_eye_top, y_eye_bot = int(h * 0.30), int(h * 0.45)
    # Left and right eye patches.
    for x_lo, x_hi in [(int(w * 0.22), int(w * 0.40)),
                       (int(w * 0.60), int(w * 0.78))]:
        out[y_eye_top:y_eye_bot, x_lo:x_hi] *= 0.65
    return out


def build_extended_fixture_bonafide(root: Path) -> Path:
    """Phase 1.5 Set B bonafide fixture.

    16 identities x 4 samples each. Each identity has a base skin-tone color
    drawn from a Fitzpatrick-inspired palette. Each image is a 64x64 RGB image
    with an oval face silhouette (Gaussian falloff from center) and darker
    eye-region patches. Per-sample noise gives 4 distinct images per identity.

    This is a procedural fixture for the synthetic cross-domain eval proxy
    -- not a substitute for real face data. See LIMITATIONS.md.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)  # different from basic fixture's seed (0)
    oval = _oval_mask(64, 64)
    eye = _eye_region_darken(64, 64)
    for identity in range(16):
        identity_dir = root / f"{identity:08d}"
        identity_dir.mkdir(exist_ok=True)
        base = np.array(_SKIN_TONE_PALETTE[identity], dtype=np.float32)
        for sample in range(4):
            # Background base * oval * eye attenuation, then per-sample noise.
            face = np.tile(base, (64, 64, 1))  # (h, w, 3)
            face = face * oval[:, :, None] * eye[:, :, None]
            # Background outside the oval falls toward neutral grey.
            # Note: oval is already applied to `face` above; using it again as the
            # alpha factor here is intentional — produces edge darkening (oval^2
            # weighting) that increases the domain gap to Set A's flat blobs.
            background = np.full((64, 64, 3), 90.0, dtype=np.float32)
            blend = oval[:, :, None]
            arr = face * blend + background * (1.0 - blend)
            noise = rng.integers(-15, 15, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(identity_dir / f"{sample}.png")
    return root


def build_fixture_dfdc(root: Path) -> Path:
    """Procedural DFDC-shaped source for tests: one chunk with 2 REAL +
    1 FAKE tiny mp4s (synthesised via ffmpeg lavfi -- no PII) and a
    matching metadata.json. Requires ffmpeg on PATH; tests should
    pytest.skip if it isn't.
    """
    import json
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    chunk = root / "chunk_00"
    chunk.mkdir(exist_ok=True)
    spec = [
        ("video_a.mp4", "REAL", None),
        ("video_b.mp4", "REAL", None),
        ("video_c.mp4", "FAKE", "video_a.mp4"),
    ]
    for name, _label, _orig in spec:
        out_path = chunk / name
        # 2-second 128x96 test pattern, h264 in mp4 (broadly compatible).
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc2=size=128x96:rate=10:d=2",
             "-pix_fmt", "yuv420p", "-c:v", "libx264",
             str(out_path)],
            check=True,
        )
    metadata = {
        name: ({"label": label, "original": orig} if orig else {"label": label})
        for name, label, orig in spec
    }
    (chunk / "metadata.json").write_text(json.dumps(metadata))
    return root


def build_fixture_real_attack(root: Path) -> Path:
    """Procedural folder-convention real-attack source for tests.

    Layout: <root>/bonafide/subjectNN/*.png and
    <root>/attack/<type>/subjectNN/*.png. Images are 96x96 RGB with
    structured noise (std well above the QC floor) so the ingester's
    resize and check_image_basic both exercise real work. No real data,
    no PII -- purely synthetic stand-ins for the capture pipeline.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260527)
    size = 96

    def _emit(class_dir: Path, n_subjects: int, base_shift: int) -> None:
        for s in range(n_subjects):
            subj = class_dir / f"subject{s:02d}"
            subj.mkdir(parents=True, exist_ok=True)
            for k in range(2):
                base = rng.integers(40, 200, size=3)
                arr = np.tile(base, (size, size, 1)).astype(np.int16)
                arr += rng.integers(-30, 30, size=(size, size, 3), dtype=np.int16)
                arr += base_shift  # per-class tint so classes are separable
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(subj / f"{k}.png")

    _emit(root / "bonafide", n_subjects=3, base_shift=0)
    _emit(root / "attack" / "print", n_subjects=3, base_shift=-25)
    _emit(root / "attack" / "replay", n_subjects=3, base_shift=25)
    return root
