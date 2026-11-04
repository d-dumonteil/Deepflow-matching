import numpy as np 
import torch
from scipy.optimize import linear_sum_assignment

B=128

def matrice_couts_torch(x0, x1):
    x0f = x0.reshape(x0.shape[0], -1).float()
    x1f = x1.reshape(x1.shape[0], -1).float()
    diff = x0f[:, None, :] - x1f[None, :, :] 
    return torch.sum(diff ** 2, dim=2) 

def appariement_ot_hongrois(x0, x1):
    x0f = x0.reshape(x0.shape[0], -1).float() 
    x1f = x1.reshape(x1.shape[0], -1).float() 
    n0 = (x0f ** 2).sum(dim=1, keepdim=True) 
    n1 = (x1f ** 2).sum(dim=1).unsqueeze(0)   
    C=n0 + n1 -2*x0f@x1f.T #broadcasting de fou 
    C=C.clamp_min(0)
    device = x0.device                 
    C_cpu = C.detach().cpu().numpy()    
    perm = linear_sum_assignment(C_cpu) 
    perm_t = torch.as_tensor(perm[1], dtype=torch.long, device=device)
    return x1[perm_t], perm_t

@torch.no_grad()
def cout_transport_batch(x0, x1, perm_t, reduction="mean"):
    C=matrice_couts_torch(x0,x1[perm_t])
    couts = C[torch.arange(C.shape[0], device=C.device), torch.arange(C.shape[0], device=C.device)] 
    return couts.mean() if reduction == "mean" else couts.sum()

def obtenir_appariement(mode="random"):
    if mode == "random":
        def _random(x0, x1):
            return x1, torch.arange(x0.shape[0], device=x0.device) #Id
        return _random
    elif mode == "ot_hongarian":
        return appariement_ot_hongrois

def main(): #tests générés par IA
    x0 = torch.randn(B, 3, 32, 32)
    x1 = torch.randn(B, 3, 32, 32)+1.0
    x1_app, perm_t = appariement_ot_hongrois(x0,x1)
    assert x1_app.shape == x1.shape
    assert torch.equal(torch.sort(perm_t).values, torch.arange(B)), "Pas une bijection"

    # Rapport OT vs random (le test métier)
    c_ot = cout_transport_batch(x0, x1, perm_t, reduction="sum").item()
    c_rand = matrice_couts_torch(x0, x1).diagonal().sum().item()   # appariement initial aléatoire
    print(f"Coût OT : {c_ot:.1f} | Coût random : {c_rand:.1f} | Rapport : {c_rand/c_ot:.2f}")

if __name__ == "__main__":
    main()