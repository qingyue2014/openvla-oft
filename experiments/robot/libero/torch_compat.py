"""PyTorch compatibility helpers for legacy LIBERO assets."""


def patch_torch_load_for_legacy_libero_assets() -> None:
    """Restore pre-PyTorch-2.6 torch.load behavior for trusted LIBERO state files."""
    import torch

    current_load = torch.load
    if getattr(current_load, "_openvla_oft_libero_legacy_patch", False):
        return

    def torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return current_load(*args, **kwargs)

    torch_load_compat._openvla_oft_libero_legacy_patch = True
    torch.load = torch_load_compat
