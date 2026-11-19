
import time
from unittest import loader

from django.core.files import images
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
np.random.seed(42)


def champ_exact(x, t, data, eps=1e-4):
    x_arr = np.asarray(x, dtype=float)
    point_unique = (x_arr.ndim == 1)                     
    xb = np.atleast_2d(x_arr) 
    data = np.asarray(data, dtype=float).reshape(-1, xb.shape[-1])  
    distance = np.sum((xb[:, None, :] - t * data[None, :, :]) ** 2, axis=2) 
    lambda_i = np.exp(-distance / (2 * max(1 - t, eps) ** 2))
    lambda_i = lambda_i / np.maximum(lambda_i.sum(axis=1, keepdims=True), 1e-12)
    moy = lambda_i @ data  
    v = (moy - xb) / max(1 - t, eps)
    return v.reshape(-1) if point_unique else v                      


def cible_efm(x0, x1, t, autres, eps=1e-4):
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    x1 = np.atleast_1d(np.asarray(x1, dtype=float))
    autres = np.asarray(autres, dtype=float).reshape(-1, x0.size)  
    x_t = (1 - t) * x0 + t * x1                                
    support = np.vstack([x1, autres])                         
    distance = np.sum((x_t - t * support) ** 2, axis=1)        
    lambda_i = np.exp(-distance / (2 * max(1 - t, eps) ** 2))
    lambda_i = lambda_i / lambda_i.sum()
    moy = lambda_i @ support                             
    return (moy - x_t) / max(1 - t, eps)
@torch.no_grad()
def cible_efm_vectorise(xt, t, support, eps=1e-2):
    t = torch.clamp(t, max=1.0 - eps) # denominateur non nul
    xt_m = xt[:, None] 
    distance = torch.sum((xt_m-t[:, None, None, None, None]*support)**2, dim = (2, 3, 4)) 
    vk = -distance/(2*(1-t[:, None])**2)
    uk = torch.softmax(vk, dim=1)
    moy = torch.sum(uk[:, :, None, None, None]*(support-xt_m)/(1-t[:, None, None, None, None]), dim = 1)
    return moy

def creer_support(x1, M, data, generator=None):
    B, C, H, W = x1.shape
    n = data.shape[0]
    generator = torch.Generator(device=x1.device)
    indices = torch.randint(0, n, (B, M - 1), device=x1.device, generator=generator)
    support = torch.empty(B, M, C, H, W, device=x1.device, dtype=x1.dtype)
    support[:, 0] = x1
    support[:,1:] = data[indices]
    return support

def efm_loss(support, eps = 1e-2, t, xt, v_theta):
    a = cible_efm_vectorise(xt, t, support, eps)
    loss = F.mse_loss(v_theta, a)
    return loss

def train_step_efm(model, x1, dataset, optimizer, M, generator=None, eps=1e-2):
    x0 = torch.randn_like(x1, generator=generator)
    t = torch.rand(x1.size(0), device=x1.device, generator=generator)
    t = torch.clamp(t, max=1.0 - eps)
    xt = (1-t[:, None, None, None])*x0 + t[:, None, None, None]*x1
    support = creer_support(x1, M, dataset, generator)
    amp_dtype = torch.float16 if x1.device.type == "cuda" else torch.bfloat16
    with torch.autocast(device=x1.device_type, dtype=amp_dtype):
        v_theta = model(xt, t)
    loss = efm_loss(support, eps, t, xt, v_theta)

    optimizer.zero_grad()
    loss.backward()
    if x1.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.optimizer.step(optimizer)
    else:
        optimizer.step()
    return loss


def preload_data(root="./data_cifar", device=None, dtype=torch.float32, batch_size=512, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
        )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = (device is not None and device.type == "cuda"))
    n, C, H, W = len(dataset), 3, 32, 32
    t = torch.empty(n, C, H, W, device = device, dtype = dtype)
    start = 0
    for images, _ in loader:  #fait avec IA
        t[start:start+len(images)] = images
        start += len(images)
    return t.to(device=device, dtype=dtype)

def update_ema(model, ema, decay=0.999):
    if ema is None:
        return
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
def train_one_epoch(model, loader, dataset_tensor, optimizer, M, device, generator=None, ema=None, use_efm=True, eps=1e-2, log_every=20):
    model.train()
    total_loss = 0.0
    n_batches = 0
    list_loss = []
    is_tpu = device.type == "xla"
    is_cuda = device.type == "cuda"
    if is_tpu:
        import torch_xla
        torch_xla.sync()
    elif is_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for (x1, labels) in loader:
        x1 = x1.to(device=device)
        if use_efm:
            loss = train_step_efm(model, x1, dataset_tensor, optimizer, M, generator, eps)
        else:
            loss = train_step(model, x1, optimizer, eps)
        total_loss += loss
        update_ema(model, ema, decay=0.999)
        if (1+n_batches) % log_every == 0:
            print(f"Batch {n_batches}, Loss: {loss.item():.4f}")
        n_batches += 1
        list_loss.append(loss.item())
    if is_tpu:
        torch_xla.sync()
    elif is_cuda:
        torch.cuda.synchronize()
    temps_epoch = time.perf_counter() - start_time
    loss_moyenne = total_loss / n_batches
    return {"loss" : loss_moyenne.item(), "temps": temps_epoch, "n_batches": n_batches}



def mesure_variance(dataset_tensor, liste_M, N_test=100, batch_size=32, device=None, generator=None, eps=1e-2):
    for M in liste_M:
        i = torch.randint(0, dataset_tensor.size(0), (N_test,), device=device, generator=generator)
        x1 = dataset_tensor[i]
        x0 = torch.randn_like(x1, generator=generator)
        t = torch.rand(N_test, device=device, generator=generator)
        t = torch.clamp(t, max=1.0 - eps)
        xt = (1-t[:, None, None, None])*x0 + t[:, None, None, None]*x1
        support = creer_support(x1, M, dataset_tensor, generator)
        a = cible_efm_vectorise(xt, t, support, eps)
        #ici calculer la variance de la loss sur les N_test échantillons je vois trop encore comment faire
        variance = 1
        return {M:variance for M in M_values}



def tirer_support(M, data):
    tirage = np.random.choice(len(data), M - 1, replace=False)
    return data[tirage]                                        


def main(): #tests générés par IA
    data = np.concatenate([np.random.normal(3, 1, size=(100, 2)),np.random.normal(-3, 1, size=(100, 2)),])
    print(data.shape)                                        

    mini = np.array([8., 10.])
    print(champ_exact(4.0, 0.0, mini))                       
    print(champ_exact(4.0, 0.5, mini))                        

    data2d = np.array([[8., 0.], [10., 0.]])
    print(champ_exact([4., 0.], 0.5, data2d))                 
    print(champ_exact([4., 2.], 0.5, data2d))             

    print(champ_exact(0.9999 * 8.0, 0.9999, mini))       

    print(cible_efm(-2.0, 8.0, 0.5, [3.0]))                   
    print(cible_efm(-2.0, 8.0, 0.999, [3.0]))                   


if __name__ == "__main__":
    main()