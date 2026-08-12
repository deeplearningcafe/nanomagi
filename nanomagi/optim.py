import torch
import omegaconf


def create_optimizer_param_groups(
    model,
    lr,
    weight_decay,
):
    param_groups = []
    no_decay_keywords = ["bias", "norm"]
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]

        if any(k in name for k in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups.append(
        {
            "params": decay,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "decay",
        }
    )
    param_groups.append(
        {
            "params": no_decay,
            "lr": lr,
            "weight_decay": 0.0,
            "name": "no_decay",
        }
    )
    return param_groups


def create_optim_scheduler(
    model,
    total_steps: int,
    conf: omegaconf.DictConfig,
    skip_warmup: bool = False,
):
    lr_val = conf.optimizer.lr
    wd_val = conf.optimizer.get("wd", 0.01)
    param_groups = create_optimizer_param_groups(
        model, lr_val, wd_val
    )
    if conf.optimizer.get("use_bitsandbytes", False):
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=lr_val,
            betas=(0.9, 0.95),
        )
    else:
        use_fused = torch.cuda.is_available()
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr_val,
            betas=(0.9, 0.95),
            fused=use_fused,
        )

    scheduler = None
    warmup_cfg = conf.optimizer.get("warmup_steps", 200)
    warmup_steps = 0 if skip_warmup else int(warmup_cfg)
    warmup_steps = (
        min(warmup_steps, total_steps - 1) if total_steps > 0 else 0
    )

    if warmup_steps > 0:
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )

    use_cos = conf.optimizer.get("use_cos_scheduler", False) or conf.get(
        "use_cos_scheduler", False
    )

    eta_min = lr_val * conf.optimizer.get("final_lr_frac", 0.05)

    if use_cos:
        print("Using cosine lr scheduler")
        cosine_steps = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=eta_min
        )
    else:
        print("Using constant lr scheduler")
        constant_steps = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=constant_steps
        )

    if warmup_steps > 0:
        return optimizer, torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [scheduler_warmup, scheduler],
            milestones=[warmup_steps],
        )

    return optimizer, scheduler
