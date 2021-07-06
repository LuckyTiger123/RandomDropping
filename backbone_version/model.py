import torch
import torch_geometric


class OurModelLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropping_method: str, backbone: str,
                 add_self_loops: bool = True, normalize: bool = True, bias: bool = True, unbias: bool = True):
        super(OurModelLayer, self).__init__()
        self.dropping_method = dropping_method

    def forward(self):
        return

