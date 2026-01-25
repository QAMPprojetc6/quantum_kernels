"""
qkernels: Quantum kernel implementations (baseline, local, multi-scale).

Unified API:
- Each kernel module exposes `build_kernel(...)` and returns (K, meta).
"""

from .feature_maps import get_feature_map_spec
from .baseline_kernel import build_kernel as build_baseline_kernel
from .baseline_kernel import build_kernel_cross as build_baseline_kernel_cross
from .local_kernel import build_kernel as build_local_kernel
from .local_kernel import build_kernel_cross as build_local_kernel_cross
from .multiscale_kernel import build_kernel as build_multiscale_kernel
from .multiscale_kernel import build_kernel_cross as build_multiscale_kernel_cross

__all__ = [
    "get_feature_map_spec",
    "build_baseline_kernel",
    "build_baseline_kernel_cross",
    "build_local_kernel",
    "build_local_kernel_cross",
    "build_multiscale_kernel",
    "build_multiscale_kernel_cross",
]
