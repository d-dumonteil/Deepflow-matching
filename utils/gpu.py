import os
import torch
import time


# --- Backend TPU n°1 : torch_xla (Kaggle, anciennes images Colab) ---
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except Exception:
    HAS_XLA = False

# --- Backend TPU n°2 : torch_tpu (images TPU récentes de Colab) ---
# Plugin PrivateUse1 -> device 'tpu', chargé automatiquement à l'import de torch.
HAS_TPU_PLUGIN = False
if not HAS_XLA:
    try:
        import torch_tpu  # noqa: F401
        HAS_TPU_PLUGIN = True
    except Exception:
        HAS_TPU_PLUGIN = False


def setup_device_and_env():
    """Détecte l'accélérateur matériel et configure les backends optimaux."""
    device, is_xla = None, False
    device_name = None

    if HAS_XLA:
        try:
            device = xm.xla_device()
            is_xla = True
        except Exception as e:
            print(f"Initialisation XLA échouée : {type(e).__name__}: {e}")
            device = None
        if is_xla:
            try:
                device_name = f"TPU XLA (type: {xm.xla_device_hw(device)})"
            except Exception:
                device_name = "TPU XLA"

    if device is None and HAS_TPU_PLUGIN:
        try:
            device = torch.device("tpu")
            torch.zeros(1, device=device)  # test réel d'allocation
            device_name = "TPU (plugin torch_tpu)"
        except Exception as e:
            print(f"Initialisation torch_tpu échouée : {type(e).__name__}: {e}")
            device = None

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
    if is_xla:
        try:
            import torch_xla
            print(f"Version torch_xla : {torch_xla.__version__}")
        except Exception:
            pass
    elif HAS_TPU_PLUGIN and device is not None and str(device.type) == "tpu":
        try:
            import torch_tpu
            print(f"Version torch_tpu : {getattr(torch_tpu, '__version__', 'inconnue')}")
        except Exception:
            pass
    elif torch.cuda.is_available():
        print(f"Version CUDA : {torch.version.cuda}")
    print("=" * 60)
    return device, is_xla


class HardwareTimer:
    def __init__(self, is_xla=False):
        self.is_xla = is_xla
        self.t0 = None

    def start(self):
        if self.is_xla:
            xm.mark_step()
        self.t0 = time.perf_counter()

    def stop(self):
        if self.is_xla:
            xm.mark_step()
        return time.perf_counter() - self.t0


def save_checkpoint(state, path, is_xla=False):
    if is_xla:
        xm.save(state, path)
    else:
        torch.save(state, path)


def xla_sync():
    import torch_xla
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        torch_xla.core.xla_model.mark_step()