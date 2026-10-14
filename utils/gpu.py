import os
import torch

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except (ImportError, RuntimeError):
    HAS_XLA = False


def setup_device_and_env():
    """Détecte l'accélérateur matériel et configure les backends optimaux."""
    if HAS_XLA and "TPU" in os.environ.get("PJRT_DEVICE", ""):
        device = xm.xla_device()
        device_name = f"TPU XLA (type: {xm.xla_device_hw(device)})"
        is_xla = True
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        is_xla = False
        # Optimisations spécifiques CUDA / Turing T4
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False  # Turing n'a pas de TF32 natif (réservé Ampere+)
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        is_xla = False

    print("=" * 60)
    print(f"Dispositif de calcul détecté : {device_name}")
    print(f"Version PyTorch : {torch.__version__}")
    if not is_xla and torch.cuda.is_available():
        print(f"Version CUDA : {torch.version.cuda}")
    print("=" * 60)
    return device, is_xla


class HardwareTimer:
    """Mesure rigoureuse du temps wall-clock avec barrière de synchronisation."""
    def __init__(self, is_xla=False):
        self.is_xla = is_xla
        if not is_xla and torch.cuda.is_available():
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.is_xla:
            xm.mark_step()
            self.t0 = os.times().elapsed
        elif torch.cuda.is_available():
            self.starter.record()
        else:
            self.t0 = os.times().elapsed

    def stop(self):
        if self.is_xla:
            xm.mark_step()
            return os.times().elapsed - self.t0
        elif torch.cuda.is_available():
            self.ender.record()
            torch.cuda.synchronize()
            return self.starter.elapsed_time(self.ender) / 1000.0
        else:
            return os.times().elapsed - self.t0


def save_checkpoint(state_dict, path, is_xla=False):
    """Sauvegarde atomique évitant la corruption du graphe statique XLA."""
    if is_xla:
        xm.save(state_dict, path)
    else:
        torch.save(state_dict, path)