import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(batch_size=128, num_workers=4, device=None):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data_cifar", train=True, download=True, transform=transform
    )

    is_cuda = device is not None and getattr(device, "type", str(device)) == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    if device is not None and "xla" in str(device):
        import torch_xla.distributed.parallel_loader as pl
        return pl.MpDeviceLoader(loader, device)

    return loader