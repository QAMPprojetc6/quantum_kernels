import inspect
import numpy as np

def _sig_ok(fn, required_params):
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    for p in required_params:
        assert p in params, f"Missing param '{p}' in {fn.__module__}.{fn.__name__}"

def test_global_kernel_signature():
    from qkernels.global_kernel import build_kernel
    _sig_ok(build_kernel, ["X","feature_map","depth","backend","seed"])
    # It does not execute (it may throw NotImplementedError)

def test_local_kernel_signature():
    from qkernels.local_kernel import build_kernel
    _sig_ok(build_kernel, ["X","feature_map","depth","backend","seed","partitions","method","agg","weights"])

def test_multiscale_kernel_signature():
    from qkernels.multiscale_kernel import build_kernel
    _sig_ok(build_kernel, ["X","feature_map","depth","backend","seed","scales","weights"])

def test_multiscale_stub_runs_raises():
    # Call the multiscale kernel if it already returns something, or if it is a stub, validate that it raises NotImplementedError.
    from qkernels.multiscale_kernel import build_kernel
    X = np.zeros((2, 4))
    try:
        build_kernel(X)
    except NotImplementedError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception: {e}")
