import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Imports spécifiques au TPU 
import torch_xla.core.xla_model as xm

from models.unet import UNetSmall
from data.dataset import get_cifar10_loader
from utils.ema import EMA
from solvers.euler import sample_euler

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()

    # 1. Sélection du device TPU
    device = xm.xla_device()
    print(f"Entraînement démarré sur l'accélérateur : {device}")

    # 2. TensorBoard local
    tensorboard_path = os.path.join(orig_cwd, cfg.logging.log_dir)
    writer = SummaryWriter(log_dir=tensorboard_path)

    # 3. Modèle et Optimiseur (pas de GradScaler requis)
    model = UNetSmall().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay
    )
    ema = EMA(model, decay=cfg.training.ema_decay)
    loader = get_cifar10_loader(batch_size=cfg.training.batch_size, device=device)

    epochs = cfg.training.epochs

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x1, _ in loader:
            x1 = x1.to(device)
            b = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_expand = t.view(b, 1, 1, 1)

            # Interpolation affine Flow Matching
            xt = (1.0 - t_expand) * x0 + t_expand * x1
            target = x1 - x0

            optimizer.zero_grad(set_to_none=True)

            # 4. Précision native bfloat16 sans GradScaler
            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                pred = model(xt, t)
                loss = torch.mean((pred - target) ** 2)

            loss.backward()

            # 5. Étape d'optimisation spécifique XLA
            xm.optimizer_step(optimizer)
            ema.update(model)

            total_loss += loss.item()

        mean_loss = total_loss / len(loader)
        print(f"Époque {epoch:02d}/{epochs:02d} | Perte CFM : {mean_loss:.4f}")

        writer.add_scalar("Loss/CFM", mean_loss, epoch)

        if epoch % cfg.logging.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = sample_euler(model, shape=(16, 3, 32, 32), steps=20, device=device)
                samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
                grid = vutils.make_grid(samples.cpu(), nrow=4)
                writer.add_image("Samples/Generated", grid, epoch)
            model.train()

    # Application EMA et sauvegarde CPU
    ema.apply_shadow(model)
    save_path = os.path.join(orig_cwd, cfg.training.save_name)
    
    # xm.save assure une écriture propre depuis la mémoire XLA
    xm.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé avec succès sous {save_path}.")
    writer.close()

if __name__ == "__main__":
    train()
