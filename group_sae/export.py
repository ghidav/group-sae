import copy
import warnings

import torch
from sae_lens import SAE, SAEConfig
from torch import nn

from . import Sae, SaeConfig


class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    # TODO: Use a fused kernel to speed up topk decoding like https://github.com/EleutherAI/sae/blob/main/sae/kernels.py
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(x)
        topk = torch.topk(x, k=self.k, dim=-1)
        values = topk.values
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


def to_sae_lens(
    sae_cfg: SaeConfig,
    sae: Sae,
    model_name: str,
    dataset_path: str,
    norm_scaling_factor: float | None = None,
    max_seq_len: int = 1024,
    hook_name: str = "blocks.0.hook_resid_post",
    hook_layer: int = 0,
    dataset_trust_remote_code: bool = True,
    dtype: str = "float32",
    device: str = "cuda",
    sae_lens_version: str = "5.3.3",
) -> SAE:
    if sae_cfg.k > 0:
        architecture = "standard"
    elif sae_cfg.k <= 0:
        if sae_cfg.jumprelu:
            architecture = "jumprelu"
        else:
            architecture = "standard"
    else:
        raise ValueError("Invalid SAE architecture")
    d_in = sae.d_in
    d_sae = sae.num_latents
    activation_fn_str = "relu" if sae_cfg.k <= 0 else "topk"
    apply_b_dec_to_input = True
    finetuning_scaling_factor = False

    # dataset it was trained on details.
    context_size = max_seq_len
    model_name = model_name
    hook_name = hook_name
    hook_layer = hook_layer
    hook_head_index = None
    prepend_bos = False
    dataset_path = dataset_path
    normalize_activations = "none"

    # misc
    dtype = dtype
    device = device
    sae_lens_cfg = SAEConfig(
        architecture=architecture,
        d_in=d_in,
        d_sae=d_sae,
        activation_fn_str=activation_fn_str,
        apply_b_dec_to_input=apply_b_dec_to_input,
        finetuning_scaling_factor=finetuning_scaling_factor,
        context_size=context_size,
        model_name=model_name,
        hook_name=hook_name,
        hook_layer=hook_layer,
        hook_head_index=hook_head_index,
        prepend_bos=prepend_bos,
        dataset_path=dataset_path,
        dataset_trust_remote_code=dataset_trust_remote_code,
        normalize_activations=normalize_activations,
        dtype=dtype,
        device=device,
        sae_lens_training_version=sae_lens_version,
        activation_fn_kwargs={} if sae_cfg.k <= 0 else {"k": sae_cfg.k},
    )
    state_dict = sae.state_dict()
    state_dict = {k: copy.deepcopy(v.cpu()) for k, v in state_dict.items()}
    state_dict["W_enc"] = state_dict.pop("encoder.weight").T.contiguous()
    state_dict["b_enc"] = state_dict.pop("encoder.bias")
    if "log_threshold" in state_dict:
        state_dict["threshold"] = torch.exp(state_dict.pop("log_threshold"))
    sae_lens_sae = SAE.from_dict(sae_lens_cfg.to_dict())
    sae_lens_sae.load_state_dict(state_dict)
    if norm_scaling_factor is None:
        warnings.warn("No norm_scaling_factor provided. Using default value of 1.0", UserWarning)
        norm_scaling_factor = 1.0
    sae_lens_sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
    if sae.cfg.jumprelu:
        # We do not apply ReLU before sending the activations to the JumpReLU function.
        sae_lens_sae.activation_fn = torch.nn.Identity()
    if sae.cfg.k > 0:
        sae_lens_sae.activation_fn = TopK(k=sae.cfg.k)
    return sae_lens_sae
