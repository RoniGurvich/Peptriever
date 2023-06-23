import torch.optim


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    cooldown: int,
    epochs: int,
    min_factor=0.001,
):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=min_factor,
        end_factor=1.0,
        total_iters=warmup,
    )
    train_epochs = epochs - warmup - cooldown
    train_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=optimizer, factor=1.0, total_iters=train_epochs
    )
    cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=min_factor,
        total_iters=cooldown,
    )
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, train_scheduler, cooldown_scheduler],
        milestones=[warmup, epochs - cooldown],
    )
    return combined_scheduler
