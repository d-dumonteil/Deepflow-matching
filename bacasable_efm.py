
import numpy as np
import torch
np.random.seed(42)


def champ_exact(x, t, data, eps=1e-4):
    x = np.atleast_1d(np.asarray(x, dtype=float))              
    data = np.asarray(data, dtype=float).reshape(-1, x.size)   
    distance = np.sum((x - t * data) ** 2, axis=1)             
    lambda_i = np.exp(-distance / (2 * max(1 - t, eps) ** 2))
    lambda_i = lambda_i / lambda_i.sum()                       
    moy = lambda_i @ data                                      
    return (moy - x) / max(1 - t, eps)                         


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