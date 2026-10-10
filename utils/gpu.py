import torch

def verify_hardware():
    assert torch.cuda.is_available(), "Erreur : CUDA requis !"
    name = torch.cuda.get_device_name(0)
    print(f"Périphérique GPU détecté : {name}")
    return name

class CUDATimer:
    def __init__(self):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
    def start(self):
        self.starter.record()
    def stop(self):
        self.ender.record()
        torch.cuda.synchronize() # Indispensable pour éviter le masquage asynchrone
        return self.starter.elapsed_time(self.ender) / 1000.0 # secondes
