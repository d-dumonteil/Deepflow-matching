import os
import torch
import time


try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except (ImportError, RuntimeError):
    HAS_XLA = False


def setup_device_and_env():
    """Détecte l'accélérateur matériel et configure les backends optimaux."""
    device, is_xla = None, False

    # TPU : test réel plutôt que la variable PJRT_DEVICE (Kaggle-only).
    # Sur Colab TPU, torch_xla est préinstallé et xm.xla_device() réussit.
    if HAS_XLA:
        try:
            device = xm.xla_device()
            is_xla = True
            device_name = f"TPU XLA (type: {xm.xla_device_hw(device)})"
        except RuntimeError:
            device = None  # torch_xla présent mais pas de TPU -> fallback

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = False  # Turing n'a pas de TF32 natif
            torch.backends.cudnn.allow_tf32 = False
        else:
            device = torch.device("cpu")
            device_name = "CPU"

    print("=" * 60)
    print(f"Dispositif de calcul détecté : {device_name}")
    print(f"Version PyTorch : {torch.__version__}")
    if not is_xla and torch.cuda.is_available():
        print(f"Version CUDA : {torch.version.cuda}")
    print("=" * 60)
    return device, is_xla

class HardwareTimer:
    def __init__(self, is_xla=False):
        self.is_xla = is_xla
        self.t0 = None
    def start(self):
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        self.t0 = time.perf_counter()
    def stop(self):
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        return time.perf_counter() - self.t0

def save_checkpoint(state, path, is_xla=False):
    if is_xla:
        import torch_xla.core.xla_model as xm
        xm.save(state, path)
    else:
        torch.save(state, path)

def xla_sync():
    import torch_xla
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        torch_xla.core.xla_model.mark_step()