"""
qkernels: Quantum kernel implementations (global, local, multi-scale).
Unified API: each module exposes `build_kernel(...)`.
"""

from .feature_maps import get_feature_map_spec
from .global_kernel import build_kernel as build_global_kernel
from .local_kernel import build_kernel as build_local_kernel
from .multiscale_kernel import build_kernel as build_multiscale_kernel

__all__ = [
    "get_feature_map_spec",
    "build_global_kernel",
    "build_local_kernel",
    "build_multiscale_kernel",
]
