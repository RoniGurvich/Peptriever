import torch
import torch.cuda


def get_device():
    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"

    device = torch.device(device_name)
    return device


def compile_model(model):
    if torch.cuda.get_device_capability()[0] >= 7:
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")
    return model
