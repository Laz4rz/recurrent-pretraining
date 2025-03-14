import torch

def list_devices_memory():
    """
    List all available devices and their memory usage.
    """
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: Free {free / (1024 ** 2):.2f} MB / {total / (1024 ** 2):.2f} MB")

