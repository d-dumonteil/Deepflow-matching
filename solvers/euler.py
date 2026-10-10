import torch

@torch.no_grad()
def sample_euler(model, shape, steps=20, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    x = torch.randn(shape, device=device) #N(0, I)
    dt = 1.0 / steps
    
    for step in range(steps):
        t_val = step * dt
        t = torch.full((shape[0],), t_val, device=device)
        v = model(x, t)
        x = x + dt * v  
        
    return x  
