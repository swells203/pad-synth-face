__version__ = "0.1.0"

# Canonical input resolution for the synthetic pipeline. Every shape-gate /
# resize / fixture emits at this size. The physics modules (print halftone,
# replay subpixel + moiré) derive their pixel-scale from the actual input
# array's shape, NOT from this constant — that decouples them from the
# global and keeps back-compat testing trivial (just pass a 64x64 input).
IMAGE_SIZE: int = 224
IMAGE_SHAPE: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3)
