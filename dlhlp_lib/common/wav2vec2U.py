# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class SamePad2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        assert len(x.size()) == 4
        if self.remove > 0:
            x = x[:, :, : -self.remove, : -self.remove]
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, lengths):
        return 0.0
    

class Discriminator(nn.Module):
    """
    Discriminator from Wav2vec2-U
    """
    def __init__(self, dim):
        super().__init__()  

        inner_dim = 256
        kernel = 6
        dilation = 1
        discriminator_causal = True
        discriminator_linear_emb: bool = True
        discriminator_depth: int = 3
        discriminator_spectral_norm: bool = False
        discriminator_weight_norm: bool = False
        discriminator_dropout: float = 0.0

        if discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=discriminator_causal),
                nn.Dropout(discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=discriminator_causal),
        ]

        if discriminator_linear_emb:
            emb_net = [make_conv(dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=discriminator_causal)
            ]
        emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(discriminator_dropout),
            *inner_net,
        )

    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        x = x.sum(dim=-1)
        x = x / x_sz
        return x
