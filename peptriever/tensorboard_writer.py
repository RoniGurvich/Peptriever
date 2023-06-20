from torch.utils.tensorboard import SummaryWriter


class Tensorboard:
    def __init__(self, output_dir, prefix: str):
        self.prefix = prefix
        self.writer = SummaryWriter(log_dir=output_dir)
        self.global_step = 0

    def write_scalars(self, scalars: dict, increment_step: bool = True):
        for metric_name, metric_value in scalars.items():
            self.writer.add_scalar(
                f"{self.prefix}/{metric_name}", metric_value, self.global_step
            )
        self.writer.flush()
        if increment_step:
            self.global_step += 1
