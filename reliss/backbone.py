"""ReLiSS Mamba encoder and its convolutional building blocks.

The module is imported by :mod:`reliss.network` and intentionally preserves
the encoder-decoder topology used for inference.

Adaptive downsampling:
    MambaEncoder accepts an optional patch_size to compute per-axis downsampling
    strides for each stage. With patch_size=None, it uses fixed strides of
    (2,2,2) for compatibility with the default behavior.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


# ============================================================
#  Adaptive stride computation
# ============================================================

def compute_adaptive_strides(
    patch_size: tuple[int, int, int],
    num_stages: int = 4,
    min_feature_size: int = 4,
) -> list[tuple[int, int, int]]:
    """Compute adaptive downsampling strides for each stage from patch_size.

    Rules:
        - Choose downsampling independently for each axis.
        - Use stride=1 when the current size is less than min_feature_size * 2.
        - Otherwise use stride=2.

    Args:
        patch_size: Spatial input patch dimensions (D, H, W).
        num_stages: Number of encoder stages.
        min_feature_size: Minimum feature-map size allowed along each axis.

    Returns:
        A list of stage strides, such as [(2,2,2), (1,2,2), ...].

    Example::

        >>> compute_adaptive_strides((128, 128, 128))
        [(2,2,2), (2,2,2), (2,2,2), (2,2,2)]
        >>> compute_adaptive_strides((48, 224, 192))
        [(2,2,2), (2,2,2), (2,2,2), (1,2,2)]
    """
    current_size = list(patch_size)
    strides = []
    for _ in range(num_stages):
        stride = []
        for d in range(3):
            if current_size[d] >= min_feature_size * 2:
                stride.append(2)
                current_size[d] = current_size[d] // 2
            else:
                stride.append(1)
        strides.append(tuple(stride))
    return strides


# ============================================================
#  Building blocks
# ============================================================

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def mamba_forward(self, x):
        B, C = x.shape[:2]
        img_dims = x.shape[2:]
        x = x.reshape(B, C, -1).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        return x.transpose(-1, -2).reshape(B, C, *img_dims)

    def forward(self, x):
        skip = x
        o1 = self.mamba_forward(x)
        o2 = self.mamba_forward(rearrange(x, "b c d w h -> b c w d h"))
        o2 = rearrange(o2, "b c w d h -> b c d w h")
        o3 = self.mamba_forward(rearrange(x, "b c d w h -> b c h w d"))
        o3 = rearrange(o3, "b c h w d -> b c d w h")
        return o1 + o2 + o3 + skip


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class GSC(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.proj  = nn.Conv3d(ch, ch, 3, 1, 1)
        self.norm  = nn.InstanceNorm3d(ch)
        self.act   = nn.ReLU()
        self.proj2 = nn.Conv3d(ch, ch, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(ch)
        self.act2  = nn.ReLU()
        self.proj3 = nn.Conv3d(ch, ch, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(ch)
        self.act3  = nn.ReLU()
        self.proj4 = nn.Conv3d(ch, ch, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(ch)
        self.act4  = nn.ReLU()

    def forward(self, x):
        res = x
        x1 = self.act(self.norm(self.proj(x)))
        x1 = self.act2(self.norm2(self.proj2(x1)))
        x2 = self.act3(self.norm3(self.proj3(x)))
        x = self.act4(self.norm4(self.proj4(x1 + x2)))
        return x + res


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, pad, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, ks, padding=pad, dilation=dilation),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)
        return x * w


class LargeKernelConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.deep = nn.Sequential(
            ConvBlock(ch, ch, 5, pad=2),
            ConvBlock(ch, ch, 3, pad=1),
            ConvBlock(ch, ch, 3, pad=2, dilation=2),
        )
        self.shortcut = nn.Sequential(
            ConvBlock(ch, ch, 3, pad=1),
            ConvBlock(ch, ch, 1, pad=0),
        )
        self.se = SEBlock(ch)

    def forward(self, x):
        return self.se(self.deep(x) + self.shortcut(x) + x)


# ============================================================
#  MambaEncoder with adaptive downsampling
# ============================================================

class MambaEncoder(nn.Module):
    """MambaEncoder backbone with adaptive downsampling.

    Args:
        in_chans: Input channels, taken from the stem's feat_size[0].
        depths: Layers per stage; defaults to [2,2,2,2].
        dims: Feature dimensions per stage; defaults to [48,96,192,384].
        patch_size: Spatial input patch dimensions (D, H, W).
            Computes independent strides per axis to avoid excessive downsampling.
            With None, uses fixed strides of (2,2,2).
        min_feature_size: Minimum feature-map size per axis; defaults to 4.
        drop_path_rate: DropPath rate; retained for API compatibility and unused.
        layer_scale_init_value: Initial LayerScale value; retained for compatibility.
        out_indices: Stages whose features are returned; defaults to [0,1,2,3].
    """

    def __init__(self, in_chans=1, depths=[2,2,2,2], dims=[48,96,192,384],
                 patch_size=None, min_feature_size=4,
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0,1,2,3]):
        super().__init__()

        num_stages = len(depths)

        # Compute downsampling strides.
        if patch_size is not None:
            self.strides = compute_adaptive_strides(
                patch_size, num_stages, min_feature_size
            )
        else:
            self.strides = [(2, 2, 2)] * num_stages

        # Downsampling layers with adaptive strides.
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv3d(dims[0], dims[0], 3, stride=self.strides[0], padding=1)))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv3d(dims[i], dims[i+1], 3, stride=self.strides[i+1], padding=1)))

        # Stages: LargeKernelConv in the first two; LargeKernelConv + Mamba in the last two.
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(4):
            self.gscs.append(nn.Sequential(*[LargeKernelConv(dims[i]) for _ in range(depths[i])]))
            if i >= 2:
                self.stages.append(nn.Sequential(*[MambaLayer(dim=dims[i]) for _ in range(depths[i])]))
            else:
                self.stages.append(None)

        # Output norms + MLPs
        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i in range(4):
            self.add_module(f'norm{i}', nn.InstanceNorm3d(dims[i]))
            self.mlps.append(MlpChannel(dims[i], 2 * dims[i]))

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            if self.stages[i] is not None:
                x = self.stages[i](x)
            if i in self.out_indices:
                x_out = self.mlps[i](getattr(self, f'norm{i}')(x))
                outs.append(x_out)
        return tuple(outs)
