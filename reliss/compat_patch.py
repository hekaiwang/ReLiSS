"""Forward-call compatibility for the inference CUDA extensions."""
from __future__ import annotations

import torch


def _patch_causal_conv1d():
    """Patch causal_conv1d_cuda to handle mamba_ssm 1.2.2 calling convention."""
    try:
        import causal_conv1d_cuda
    except ImportError:
        return  # 没装就不管

    _orig_fwd = causal_conv1d_cuda.causal_conv1d_fwd

    # 检测是否需要 patch: 尝试用 7 参数调用 fwd
    # 如果原生支持 7 参数就不需要 patch
    try:
        # 用小 tensor 测试
        _test_x = torch.zeros(1, 1, 4, device="cpu")
        _test_w = torch.zeros(1, 3, device="cpu")
        _orig_fwd(_test_x, _test_w, None, None, None, None, True)
        return  # 原生支持 7 参数，不需要 patch
    except TypeError:
        pass  # 需要 patch
    except Exception:
        pass  # 其他错误（如 CUDA 不可用），继续 patch

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
