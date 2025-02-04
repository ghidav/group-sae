import argparse
import os
import re
import warnings
from importlib.metadata import version

import torch

from group_sae.export import Sae, SaeConfig, to_sae_lens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path of the folder containing the SAE model and the configuration file. "
        "For example, 'path/to/checkpoints/EleutherAI/pythia-160m-deduped/step_4999/layers.3/'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory to save the SAE lens model."
        "If left unspecified, the SAELens SAE will be saved in the model_path under the `sae_lens` folder.",
    )
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument(
        "--norm_scaling_factors_path",
        type=str,
        default=None,
        help="Path to the norm scaling factors. "
        "By default, the norm scaling factor are found in the checkpoint folder, e.g.: "
        "'path/to/checkpoints/EleutherAI/pythia-160m-deduped/step_4999/scaling_factors.pt'."
        " If left unspecified, the norm scaling factor will be set to 1.0. ",
    )
    parser.add_argument(
        "--hook_name",
        type=str,
        required=True,
        help="The name of the hook in SAELens format. For example, 'blocks.0.hook_resid_post'.",
    )
    parser.add_argument(
        "--hook_layer",
        type=int,
        required=True,
        help="The layer of the hook. It must be an integer and equal to the layer of the hook.",
    )
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sae_lens_version", type=str, default="5.3.0")
    args = parser.parse_args()

    layer_in_hook_name = re.findall(r"\d+", args.hook_name)[0]
    if args.hook_layer != int(layer_in_hook_name):
        raise ValueError(
            f"The hook layer {args.hook_layer} does not match the layer in the hook name {layer_in_hook_name}."
        )

    current_sae_lens_version = version("sae_lens")
    if current_sae_lens_version != args.sae_lens_version:
        warnings.warn(
            f"The current version of sae_lens is {current_sae_lens_version}, "
            f"while the specified version is {args.sae_lens_version}."
            "Beware of potential compatibility issues."
        )

    sae = Sae.load_from_disk(args.model_path)
    sae_cfg = SaeConfig.load_json(os.path.join(args.model_path, "cfg.json"))
    norm_scaling_factor = None
    if args.norm_scaling_factors_path is not None:
        hookpoint = args.model_path.split(os.sep)[-1]
        norm_scaling_factors = torch.load(args.norm_scaling_factors_path)
        norm_scaling_factor = norm_scaling_factors[hookpoint]
    sae_lens = to_sae_lens(
        sae=sae,
        sae_cfg=sae_cfg,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        norm_scaling_factor=norm_scaling_factor,
        max_seq_len=args.max_seq_len,
        hook_name=args.hook_name,
        hook_layer=args.hook_layer,
        dtype=args.dtype,
        device=args.device,
        sae_lens_version=args.sae_lens_version,
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.model_path, "sae_lens")
    os.makedirs(output_dir, exist_ok=True)
    sae_lens.save_model(output_dir)

    # Test the forward pass of the SAE and SAE lens
    tensor = torch.rand(1, sae.d_in, device=args.device, dtype=torch.float32)

    sae_lens.to(args.device)
    sae_lens_out = sae_lens.forward(tensor)

    if norm_scaling_factor is None:
        warnings.warn("No norm scaling factor provided, setting it to 1.0")
        norm_scaling_factor = 1.0
    sae.to(args.device)
    sae_out = sae.forward(tensor * norm_scaling_factor).sae_out / norm_scaling_factor

    torch.testing.assert_close(sae_lens_out, sae_out)
