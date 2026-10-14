import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from models.unet import UNetSmall
from data.dataset import get_cifar10_loader
from utils.ema import EMA
from utils.hardware import setup_device_and_env, HardwareTimer, save_checkpoint
from solvers.ode import sample_euler


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    seed = cfg.training.get("seed", 42)
    seed_everything(seed)

    orig_cwd = hydra.utils.get_original_cwd()
    device, is_xla = setup_device_and_env()
    timer = HardwareTimer(is_xla=is_xla)

    tensorboard_path = os.path.join(orig_cwd, cfg.logging.log_dir, f"seed_{seed}")
    writer = SummaryWriter(log_dir=tensorboard_path)

    model = UNetSmall(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    ema = EMA(model, decay=cfg.training.ema_decay)
    loader = get_cifar10_loader(batch_size=cfg.training.batch_size, device=device)

    # Précision mixte : native bfloat16 sur TPU, fp16 + GradScaler sur GPU T4
    use_cuda_amp = (not is_xla) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    amp_dtype = torch.bfloat16 if is_xla else torch.float16
    amp_device = "xla" if is_xla else "cuda"

    epochs = cfg.training.epochs
    best_loss = float("inf")

    print(f"Démarrage de l'entraînement (graine {seed}) pour {epochs} époques...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        timer.start()

        for x1, _ in loader:
            x1 = x1.to(device, non_blocking=True)
            b = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_expand = t.view(b, 1, 1, 1)

            xt = (1.0 - t_expand) * x0 + t_expand * x1
            target = x1 - x0

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=(is_xla or use_cuda_amp)):
                pred = model(xt, t)
                loss = torch.mean((pred - target) ** 2)

            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if is_xla:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

            ema.update(model)
            total_loss += loss.item()

        epoch_time = timer.stop()
        mean_loss = total_loss / len(loader)

        print(f"Époque {epoch:02d}/{epochs:02d} | Perte : {mean_loss:.4f} | Temps : {epoch_time:.2f} s")
        writer.add_scalar("Loss/CFM", mean_loss, epoch)
        writer.add_scalar("Time/Epoch", epoch_time, epoch)

        # Checkpointage régulier et reprise (toutes les 10 époques ou meilleure loss)
        if epoch % 10 == 0 or mean_loss < best_loss:
            if mean_loss < best_loss:
                best_loss = mean_loss
            ckpt_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "seed": seed,
                "loss": mean_loss,
            }
            ckpt_path = os.path.join(orig_cwd, f"checkpoint_seed_{seed}_ep{epoch}.pt")
            save_checkpoint(ckpt_state, ckpt_path, is_xla=is_xla)

        if epoch % cfg.logging.sample_every == 0:
            ema.apply_shadow(model)
            model.eval()
            with torch.no_grad():
                samples = sample_euler(model, shape=(16, 3, 32, 32), steps=20, device=device)
                samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
                grid = vutils.make_grid(samples.cpu(), nrow=4)
                writer.add_image("Samples/Generated", grid, epoch)
            ema.restore(model)

    # Sauvegarde finale avec application stricte des poids EMA
    ema.apply_shadow(model)
    final_path = os.path.join(orig_cwd, cfg.training.save_name)
    save_checkpoint(model.state_dict(), final_path, is_xla=is_xla)
    print(f"Modèle EMA final sauvegardé sous {final_path}.")
    writer.close()


if __name__ == "__main__":
    train()