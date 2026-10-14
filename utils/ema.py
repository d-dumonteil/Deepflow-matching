import torch

class EMA:
    def __init__(self, model, decay=0.999, beta=None):
        self.decay = beta if beta is not None else decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        params = [param.data for name, param in model.named_parameters() if param.requires_grad]
        shadows = [self.shadow[name] for name, param in model.named_parameters() if param.requires_grad]
        # Opération vectorisée groupée (rapide sur GPU T4)
        torch._foreach_lerp_(shadows, params, 1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])