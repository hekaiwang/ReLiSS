from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from . import compat_patch  # noqa: F401
from .backbone import MambaEncoder, compute_adaptive_strides


class ReLiSSSpectralAvailabilityEstimator(nn.Module):
    """Frequency-derived availability-conditioned modality score estimator.

    The estimator exposes three distinct quantities for every modality: raw
    MLP logits, their sigmoid scores, and a deterministic exact-zero presence
    mask. Applying the hard presence mask is deliberately left to the network
    so supervision can act directly on the unmasked logits.
    """

    def __init__(
        self,
        in_channels: int,
        freq_ratio: float = 0.25,
        min_mid: int = 16,
        feature_mode: str = "spectral",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.freq_ratio = float(freq_ratio)
        if feature_mode != "spectral":
            raise ValueError(f"unsupported availability feature mode: {feature_mode}")
        self.feature_mode = feature_mode
        feat_per_channel = 6
        feat_dim = self.in_channels * feat_per_channel
        mid = max(feat_dim, min_mid)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, self.in_channels),
        )
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def _spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        fft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        amp = torch.abs(fft)

        fd = torch.fft.fftfreq(d, device=x.device)
        fh = torch.fft.fftfreq(h, device=x.device)
        fw = torch.fft.rfftfreq(w, device=x.device)
        gd, gh, gw = torch.meshgrid(fd, fh, fw, indexing="ij")
        radius = torch.sqrt(gd.square() + gh.square() + gw.square())
        low_mask = radius <= self.freq_ratio
        high_mask = ~low_mask

        total_energy = amp.sum(dim=(-3, -2, -1)) + 1e-8
        low_energy = (amp * low_mask.view(1, 1, d, h, -1)).sum(dim=(-3, -2, -1))
        high_energy = (amp * high_mask.view(1, 1, d, h, -1)).sum(dim=(-3, -2, -1))
        amp_flat = amp.reshape(b, c, -1)

        spatial_abs = x.float().abs().mean(dim=(-3, -2, -1))
        global_energy = torch.log(amp.mean(dim=(-3, -2, -1)) + 1e-6)
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        spectral_cv = amp_flat.std(dim=-1) / (amp_flat.mean(dim=-1) + 1e-6)
        low_high_log_ratio = torch.log(low_energy / (high_energy + 1e-8) + 1e-6)

        return torch.cat(
            [
                spatial_abs,
                global_energy,
                low_ratio,
                high_ratio,
                spectral_cv,
                low_high_log_ratio,
            ],
            dim=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._spectral_features(x)
        logits = self.mlp(features)
        raw_score = torch.sigmoid(logits)
        present = (x.float().abs().mean(dim=(-3, -2, -1)) > 1e-7).to(raw_score.dtype)
        return logits, raw_score, present


class ReliabilityFiLM(nn.Module):
    """Identity-initialized reliability-conditioned affine modulation."""

    def __init__(self, reliability_channels: int, feature_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        mid = max(hidden, reliability_channels * 4)
        self.net = nn.Sequential(
            nn.Linear(reliability_channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 2 * self.feature_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(reliability.float()).to(dtype=z.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(z.shape[0], self.feature_channels, 1, 1, 1)
        beta = beta.view(z.shape[0], self.feature_channels, 1, 1, 1)
        return z * (1.0 + gamma) + beta


class ReLiSSUNet(nn.Module):
    """Availability-conditioned ReLiSS network for missing modalities.

    Spectral scores condition modality fusion and multi-scale FiLM. An exact-zero
    availability mask removes absent modalities from fusion.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depths: list[int] | None = None,
        feat_size: list[int] | None = None,
        deep_supervision: bool = False,
        norm_name: str = "instance",
        patch_size: tuple[int, int, int] | None = None,
        drop_prob: float = 0.0,
        fusion_mode: str = "learned",
        availability_feature_mode: str = "spectral",
        enable_reliability_film: bool = True,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if feat_size is None:
            feat_size = [48, 96, 192, 384]
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.feat_size = [int(i) for i in feat_size]
        if deep_supervision:
            raise ValueError("The public ReLiSS package exposes inference outputs only")
        self.deep_supervision = False
        self.patch_size = patch_size
        if fusion_mode != "learned" or availability_feature_mode != "spectral" or not enable_reliability_film:
            raise ValueError("ReLiSS uses learned spectral fusion with reliability FiLM")
        self.fusion_mode = fusion_mode
        self.availability_feature_mode = availability_feature_mode
        self.enable_reliability_film = bool(enable_reliability_film)
        self.hidden_size = 2 * self.feat_size[-1]

        if patch_size is not None:
            strides = compute_adaptive_strides(patch_size, num_stages=4)
        else:
            strides = [(2, 2, 2)] * 4
        self.strides = strides

        self.reliability_estimator = ReLiSSSpectralAvailabilityEstimator(
            self.in_channels, feature_mode=self.availability_feature_mode,
        )
        initial_sharpness = torch.tensor(3.0, dtype=torch.float32)
        initial_raw_sharpness = torch.log(torch.expm1(initial_sharpness - 0.5))
        self.fusion_sharpness_raw = nn.Parameter(initial_raw_sharpness)
        self.modality_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(1, self.feat_size[0], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(self.feat_size[0]),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.in_channels)
            ]
        )

        conditioning = lambda channels: ReliabilityFiLM(self.in_channels, channels)
        self.condition_stem = conditioning(self.feat_size[0])
        self.condition_outs = nn.ModuleList(
            [conditioning(channels) for channels in self.feat_size]
        )
        self.condition_hidden = conditioning(self.hidden_size)

        self.encoder = MambaEncoder(
            in_chans=self.feat_size[0],
            depths=[int(i) for i in depths],
            dims=self.feat_size,
            patch_size=patch_size,
        )

        spatial_dims = 3
        res_block = True
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0],
            out_channels=self.feat_size[0], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0],
            out_channels=self.feat_size[1], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1],
            out_channels=self.feat_size[2], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2],
            out_channels=self.feat_size[3], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3],
            out_channels=self.hidden_size, kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.hidden_size,
            out_channels=self.feat_size[3], kernel_size=3,
            upsample_kernel_size=strides[3],
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3],
            out_channels=self.feat_size[2], kernel_size=3,
            upsample_kernel_size=strides[2],
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2],
            out_channels=self.feat_size[1], kernel_size=3,
            upsample_kernel_size=strides[1],
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1],
            out_channels=self.feat_size[0], kernel_size=3,
            upsample_kernel_size=strides[0],
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0],
            out_channels=self.feat_size[0], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )

        self.dropout = nn.Dropout3d(p=drop_prob) if drop_prob > 0 else None
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0],
            out_channels=self.num_classes,
        )
        self.ds_head1 = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1],
            out_channels=self.num_classes,
        )
        self.ds_head2 = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2],
            out_channels=self.num_classes,
        )
        self.ds_head3 = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3],
            out_channels=self.num_classes,
        )

        self.current_reliability_logits: torch.Tensor | None = None
        self.current_raw_reliability: torch.Tensor | None = None
        self.current_availability: torch.Tensor | None = None
        self.current_reliability: torch.Tensor | None = None
        self.last_reliability: torch.Tensor | None = None
        self.last_fusion_weights: torch.Tensor | None = None

    def _reliability_fused_stem(
        self,
        x: torch.Tensor,
        reliability: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        per_modality = []
        for idx, stem in enumerate(self.modality_stems):
            mod_x = x[:, idx:idx + 1] * reliability[:, idx].view(x.shape[0], 1, 1, 1, 1).to(x.dtype)
            per_modality.append(stem(mod_x))
        stacked = torch.stack(per_modality, dim=1)

        sharpness = 0.5 + F.softplus(self.fusion_sharpness_raw)
        fusion_logits = sharpness * reliability.float()
        present = availability.to(dtype=torch.bool)
        any_present = present.any(dim=1, keepdim=True)
        masked_logits = fusion_logits.masked_fill(~present, torch.finfo(fusion_logits.dtype).min)
        safe_logits = torch.where(any_present, masked_logits, torch.zeros_like(masked_logits))
        fusion_weights = torch.softmax(safe_logits, dim=1)
        fusion_weights = fusion_weights * present.to(dtype=fusion_weights.dtype)
        fusion_weights = torch.where(any_present, fusion_weights, torch.zeros_like(fusion_weights))
        fusion_weights = fusion_weights.to(dtype=x.dtype)
        self.last_fusion_weights = fusion_weights.detach()
        fused = (stacked * fusion_weights.view(x.shape[0], self.in_channels, 1, 1, 1, 1)).sum(dim=1)
        conditioned = self.condition_stem(fused, reliability)
        return conditioned * any_present.view(x.shape[0], 1, 1, 1, 1).to(conditioned.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        reliability_logits, raw_reliability, availability = self.reliability_estimator(x)
        reliability = availability * raw_reliability

        self.current_reliability_logits = reliability_logits
        self.current_raw_reliability = raw_reliability
        self.current_availability = availability
        self.current_reliability = reliability
        self.last_reliability = reliability.detach()

        x_stem = self._reliability_fused_stem(x, reliability, availability)
        enc1 = self.encoder1(x_stem)

        outs = list(self.encoder(x_stem))
        outs = [condition(out, reliability) for condition, out in zip(self.condition_outs, outs)]

        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])

        enc_hidden = self.condition_hidden(self.encoder5(outs[3]), reliability)
        if self.dropout is not None:
            enc_hidden = self.dropout(enc_hidden)

        dec3 = self.decoder5(enc_hidden, enc4)
        if self.dropout is not None:
            dec3 = self.dropout(dec3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        full_res = self.out(out)
        return full_res
