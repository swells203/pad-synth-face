import numpy as np

from pad_synth_face.attacks.replay import _moire, _subpixel_grid


def test_subpixel_pitch_back_compat_at_64():
    """At 64x64 the column-stripe pattern repeats every 3 columns (the
    pre-bump pitch). The first three columns of the leftmost row must
    contain the three known relative levels [0.92, 0.96, 0.90]."""
    pattern = _subpixel_grid(64, 64)
    row0 = pattern[0, :3, 0]
    assert np.allclose(row0, [0.92, 0.96, 0.90])


def test_subpixel_pitch_scales_with_image_dim():
    """At 224x224 the same column-stripe pattern occupies ~11 columns per
    repeat (round(224/64 * 3)) -- same visible angular size."""
    pattern = _subpixel_grid(224, 224)
    row0 = pattern[0, :, 0]
    # Find the first column index >= 1 where value 0.92 reappears -- that's
    # the pitch.
    first_period = None
    for k in range(1, len(row0)):
        if np.isclose(row0[k], 0.92):
            first_period = k
            break
    assert first_period == 11, f"expected pitch 11 at 224x224, got {first_period}"


def test_moire_freq_back_compat_at_64():
    """At 64x64 with refresh_hz=60 the moiré freq equals the pre-bump
    0.18 cycles/pixel. Detect via dominant FFT peak in a center-row slice."""
    rng = np.random.default_rng(0)
    pat = _moire(64, 64, refresh_hz=60, rng=rng)
    row = pat[32, :, 0]
    spec = np.abs(np.fft.rfft(row - row.mean()))
    peak_k = int(np.argmax(spec[1:])) + 1
    # 64 pixels * 0.18 cycles/pixel ≈ 11.5 cycles ≈ peak at k=11 or k=12.
    assert peak_k in (11, 12), f"expected k in 11..12 at 64x64, got {peak_k}"


def test_moire_freq_scales_with_image_dim():
    """At 224x224 the moiré freq is scaled by 64/224 so the bands-per-image
    count stays the same as at 64."""
    rng = np.random.default_rng(0)
    pat = _moire(224, 224, refresh_hz=60, rng=rng)
    row = pat[112, :, 0]
    spec = np.abs(np.fft.rfft(row - row.mean()))
    peak_k = int(np.argmax(spec[1:])) + 1
    # Same 11..12 cycles per image-width (NOT per pixel) at any resolution.
    assert peak_k in (11, 12), f"expected k in 11..12 at 224x224, got {peak_k}"
