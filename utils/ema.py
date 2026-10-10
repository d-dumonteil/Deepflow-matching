import torch

class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.beta).add_(param.data, alpha=1.0 - self.beta)

    def apply_shadow(self, model):
        self.backup = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
