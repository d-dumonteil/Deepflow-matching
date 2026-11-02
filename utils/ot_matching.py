import numpy as np 

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except (ImportError, RuntimeError):
    HAS_XLA = False

B=128

def matrice_couts_torch(x0, x1):
    x0f = x0.reshape(x0.shape[0], -1).float()
    x1f = x1.reshape(x1.shape[0], -1).float()
    diff = x0f[:, None, :] - x1f[None, :, :] 
    return torch.sum(diff ** 2, dim=2) 

def appariement_ot_hongrois(x0, x1, sur_device=False):
    x0f = x0.reshape(B, -1).float() 
    x1f = x1.reshape(B, -1).float() 
    n0 = (x0f ** 2).sum(dim=1, keepdim=True) 
    n1 = (x1f ** 2).sum(dim=1).unsqueeze(0)   
    C=n0 + n1 -2*x0f@x1f.T #broadcasting de fou 
    device = x0.device                 
    C_cpu = C.detach().cpu().numpy()    
    perm = linear_sum_assignment(C_cpu) 
    perm_t = torch.as_tensor(perm[1], dtype=torch.long, device=device)
    return x1[perm_t]  


def cout_transport_batch(x0, x1, reduction="mean"):
    x1_appariement=appariement_ot_hongrois(x0,x1)
    C=matrice_couts_torch(x0,x1_appariement)
    return np.sum(C[i,i] for i in range(len(x0)))

def obtenir_appariement(mode="random"):
    if mode == "random":
        return lambda x0, x1: x1           
    elif mode == "ot_hongarian":
        def _apparie(x0, x1):
            return appariement_ot_hongrois(x0, x1)
        return _apparie
    return 

def main():
    x0 = torch.randn(B, 3, 32, 32)
    x1 = torch.randn(B, 3, 32, 32)+1.0
    x1_appariement = appariement_ot_hongrois(x0,x1)
    assert(x1_appariement.shape==x1.shape)
    return 

if __name__ == "__main__":
    main()