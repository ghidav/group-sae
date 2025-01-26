from dataclasses import dataclass
from typing import cast

import einops
import torch
import torch.nn as nn

from group_sae.utils import JumpReLU


@dataclass
class SAEConfig:
    d_in: int
    c: int


@dataclass
class TopKSAEConfig(SAEConfig):
    k: int


@dataclass
class JumpReLUSAEConfig(SAEConfig):
    lam: float


class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super(TopKSAE, self).__init__()

        self.d_sae = cfg.d_in * cfg.c
        self.W_enc = nn.Parameter(torch.randn(cfg.d_in, self.d_sae))
        torch.nn.init.kaiming_uniform_(self.W_enc)
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))

        self.W_dec = nn.Parameter(torch.randn(self.d_sae, cfg.d_in))
        self.W_dec.data = self.W_enc.data.T.clone()
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))

        self.set_decoder_norm_to_unit_norm()

    def forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )


class TopKSAE(SAE):
    def __init__(self, cfg: TopKSAEConfig):
        super(TopKSAE, self).__init__(cfg)

        self.k = cfg.k

    def forward(self, x):
        h = torch.einsum("d_in d_sae, d_in -> d_sae", self.W_enc, x - self.b_dec) + self.b_enc
        h = torch.relu(h)
        top_acts, top_ids = h.topk(self.k, sorted=False)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.W_dec.shape[-1],))
        acts = buf.scatter_(dim=-1, index=top_ids, src=top_acts)
        return torch.einsum("d_sae d_in, d_sae -> d_in", self.W_dec, acts) + self.b_dec


class JumpReLUSAE(SAE):
    def __init__(self, cfg: JumpReLUSAEConfig):
        super(JumpReLUSAE, self).__init__(cfg)

        self.log_threshold = nn.Parameter(torch.ones(self.num_latents) * torch.log(0.001))
        self.bandwidth = 0.001

    def forward(self, x):
        h = torch.einsum("d_in d_sae, d_in -> d_sae", self.W_enc, x - self.b_dec) + self.b_enc
        threshold = torch.exp(self.log_threshold)
        h = cast(torch.Tensor, JumpReLU.apply(h, threshold, self.bandwidth))
        return torch.einsum("d_sae d_in, d_sae -> d_in", self.W_dec, h) + self.b_dec
