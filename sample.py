import torch
import torchvision.utils as vutils
from models.unet import UNetSmall
from solvers.euler import sample_euler
from utils.gpu import setup_device_and_env 
device, _ = setup_device_and_env()

model = UNetSmall().to(device)
checkpoint = torch.load("model_baseline.pt", map_location=device, weights_only=True)
state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
model.load_state_dict(state)
model.eval()

print(f"Génération de 16 échantillons sur {device}...")
with torch.no_grad():
    x = sample_euler(model, shape=(16, 3, 32, 32), steps=20, device=device)
    # Dénormalisation [-1, 1] -> [0, 1]
    images = (x.clamp(-1.0, 1.0) + 1.0) / 2.0

vutils.save_image(images.cpu(), "samples_generated.png", nrow=4)
print("Images générées enregistrées dans samples_generated.png")
