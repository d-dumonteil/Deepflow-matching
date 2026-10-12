import torch
from torchdiffeq import odeint

@torch.no_grad()
def sample_euler(model, shape, steps=20, device=None):
    """Solveur d'Euler explicite vectorisé."""
    if device is None:
        device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for step in range(steps):
        t_val = step * dt
        t = torch.full((shape[0],), t_val, device=device)
        v = model(x, t)
        x = x + dt * v
    return x


@torch.no_grad()
def sample_ode(model, shape, method="euler", steps=20, device=None, atol=1e-5, rtol=1e-5):
    """Solveur généralisé interfacé avec torchdiffeq (rk4, dopri5, euler)."""
    if device is None:
        device = next(model.parameters()).device
    x0 = torch.randn(shape, device=device)

    class ODEWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, t, x):
            t_vec = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            return self.net(x, t_vec)

    ode_fn = ODEWrapper(model)
    t_span = torch.linspace(0.0, 1.0, steps + 1, device=device)

    if method == "euler":
        trajectory = odeint(ode_fn, x0, t_span, method="euler")
    elif method == "rk4":
        trajectory = odeint(ode_fn, x0, t_span, method="rk4")
    elif method == "dopri5":
        t_eval = torch.tensor([0.0, 1.0], device=device)
        trajectory = odeint(ode_fn, x0, t_eval, method="dopri5", atol=atol, rtol=rtol)
    else:
        raise ValueError(f"Méthode {method} non supportée.")

    return trajectory[-1]