"""Apply exact-zero missingness at the network input after preprocessing."""
import torch
from torch import nn

class MissingModalityWrapper(nn.Module):
    def __init__(self, network, channels, keep):
        super().__init__()
        self.network = network
        mask = torch.zeros(1, channels, 1, 1, 1)
        mask[:, keep] = 1
        self.register_buffer('availability', mask, persistent=False)

    def forward(self, x):
        return self.network(x * self.availability.to(dtype=x.dtype))


def normalize_network_state(state):
    """Remove PyTorch wrapper prefixes and reject ambiguous parameter names."""
    result = {}
    for key, value in state.items():
        while key.startswith(("_orig_mod.", "module.")):
            key = key.split(".", 1)[1]
        if key in result:
            raise ValueError(f"Duplicate checkpoint parameter: {key}")
        result[key] = value
    return result


def build_inference_network(plans, dataset, configuration, state):
    """Construct the published segmentation model and strictly load its parameters."""
    from .network import ReLiSSUNet
    state = normalize_network_state(state)
    network = ReLiSSUNet(
        in_channels=len(dataset["channel_names"]),
        num_classes=int(plans.get_label_manager(dataset).num_segmentation_heads),
        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
        patch_size=tuple(configuration.patch_size), deep_supervision=False,
    )
    network.load_state_dict(state, strict=True)
    return network.eval()
