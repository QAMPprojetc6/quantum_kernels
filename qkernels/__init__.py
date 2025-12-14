"""
qkernels: Quantum kernel implementations (baseline, local, multi-scale).

Unified API:
- Each kernel module exposes `build_kernel(...)` and returns (K, meta).
"""

from .feature_maps import get_feature_map_spec
from .baseline_kernel import build_kernel as build_baseline_kernel
from .local_kernel import build_kernel as build_local_kernel
from .multiscale_kernel import build_kernel as build_multiscale_kernel

__all__ = [
    "get_feature_map_spec",
    "build_baseline_kernel",
    "build_local_kernel",
    "build_multiscale_kernel",
]
