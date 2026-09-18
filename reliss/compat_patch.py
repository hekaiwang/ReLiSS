"""Forward-call compatibility for the inference CUDA extensions."""
from __future__ import annotations

import torch


def _patch_causal_conv1d():
    """Patch causal_conv1d_cuda to handle mamba_ssm 1.2.2 calling convention."""
    try:
        import causal_conv1d_cuda
    except ImportError:
        return  # Skip when the extension is not installed.

    _orig_fwd = causal_conv1d_cuda.causal_conv1d_fwd

    # Probe the seven-argument forward signature.
    # No patch is needed if the extension supports it.
    try:
        # Test with small tensors.
        _test_x = torch.zeros(1, 1, 4, device="cpu")
        _test_w = torch.zeros(1, 3, device="cpu")
        _orig_fwd(_test_x, _test_w, None, None, None, None, True)
        return  # The seven-argument signature is supported.
    except TypeError:
        pass  # Adapt the unsupported signature.
    except Exception:
        pass  # Continue patching after other errors, such as unavailable CUDA.

    def _patched_fwd(*args):
        """Wrap fwd: 7-arg (mamba_ssm 1.2.2) → 5-arg (causal_conv1d 1.1.x)."""
        if len(args) == 7:
            # mamba_ssm 1.2.2: (x, weight, bias, seq_idx, initial_states, final_states_out, silu_activation)
            # causal_conv1d 1.1.x: (x, weight, bias, seq_idx, silu_activation)
            x, weight, bias, seq_idx, _initial_states, _final_states_out, silu_activation = args
            return _orig_fwd(x, weight, bias, seq_idx, silu_activation)
        return _orig_fwd(*args)

    causal_conv1d_cuda.causal_conv1d_fwd = _patched_fwd


_patch_causal_conv1d()
