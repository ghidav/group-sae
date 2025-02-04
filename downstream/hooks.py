import torch


class PassThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_hat):
        # Forward: same as "x_hat + (x - x_hat).detach()"
        # Numerically, this equals x, but ensures dictionary gradient is not canceled.
        return x_hat + (x - x_hat).detach()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: pass the gradient as if x_recon == x.
        # That means ∂(x_recon)/∂x = I and ∂(x_recon)/∂x_hat = I as well
        # so each gets grad_output.
        #
        # If you wanted to treat x_hat differently, you could modify
        # the second returned value, but here we "split" the gradient equally.
        return grad_output, grad_output


def sae_hook_pass_through(act, hook, sae, cache):
    original_shape = act.shape
    if len(original_shape) == 4:
        x = act.reshape(act.shape[0], act.shape[1], -1).clone()
    else:
        x = act.clone()

    f = sae.encode(x)
    x_hat = sae.decode(f)

    if torch.is_grad_enabled():
        f.retain_grad()
    cache[hook.name] = f

    x_recon = PassThrough.apply(x, x_hat)
    if len(original_shape) == 4:
        return x_recon.reshape(original_shape)
    return x_recon


def sae_hook(act, hook, sae, cache):
    original_shape = act.shape
    if len(original_shape) == 4:
        x = act.reshape(act.shape[0], act.shape[1], -1).clone()
    else:
        x = act.clone()

    f = sae.encode(x)
    x_hat = sae.decode(f)

    if torch.is_grad_enabled():
        f.retain_grad()

    residual = x - x_hat
    cache[hook.name] = f

    x_recon = x_hat + residual.detach()

    if len(original_shape) == 4:
        return x_recon.reshape(original_shape)

    return x_recon


def sae_ig_patching_hook(act, hook, sae, patch, cache):

    patch.retain_grad()
    x_hat = sae.decode(patch)

    # Cache the computed activations and residuals
    if hook.name not in cache:
        cache[hook.name] = patch
    else:
        raise ValueError("Patching hook should only be called once per hook.")

    if len(act.shape) == 4:
        x_recon = x_hat.reshape(act.shape)
    else:
        x_recon = x_hat

    return x_recon


@torch.no_grad()
def sae_features_hook(
    x, hook, sae, feature_mask, feature_avg, resid=False, ablation="faithfulness", cache=None
) -> torch.Tensor:
    original_shape = x.shape  # (batch, seq, d_model)

    if len(original_shape) == 4:
        x = x.reshape(x.shape[0], x.shape[1], -1).clone()
    else:
        x = x.clone()

    sae_features = sae.encode(x)  # (batch, seq, d_sae)

    if resid:
        full_recon = sae.decode(sae_features)
        resid_ = x - full_recon

    feature_avg = feature_avg.to(x.device)
    feature_mask = feature_mask.to(x.device)
    if ablation == "empty":
        sae_features = feature_avg
    elif ablation == "faithfulness":
        sae_features = sae_features * feature_mask + feature_avg * ~feature_mask
    elif ablation == "completeness":
        sae_features = sae_features * ~feature_mask + feature_avg * feature_mask
    else:
        raise NotImplementedError

    if cache is not None:
        if hook.name in cache:
            cache[hook.name].append(sae_features.cpu())
        else:
            cache[hook.name] = [sae_features.cpu()]

    x_recon = sae.decode(sae_features)

    if len(original_shape) == 4:
        return x_recon.reshape(original_shape)

    if resid:
        return x_recon + resid_
    else:
        return x_recon.to(x.device)
