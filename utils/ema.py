import torch

class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        self.param_keys = [p for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self, model):
        params = [p.data for p in self.param_keys]
        # lerp_(target, weight) fait self = self + weight * (target - self)
        # soit shadow = shadow * beta + target * (1 - beta)
        torch._foreach_lerp_(self.shadow, params, 1.0 - self.beta)

    def apply_shadow(self, model):
        self.backup = [p.clone().detach() for p in self.param_keys]
        for p, s in zip(self.param_keys, self.shadow):
            p.data.copy_(s)

    def restore(self, model):
        for p, b in zip(self.param_keys, self.backup):
            p.data.copy_(b)