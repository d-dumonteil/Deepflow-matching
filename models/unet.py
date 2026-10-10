import math
import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        indices = torch.arange(half_dim, device=device).float()

        facteur_echelle = math.log(10000.0) / (half_dim - 1)
        frequences = torch.exp(-indices * facteur_echelle)

        t_colonne = t.unsqueeze(1) #[B, 1]

        frequences_ligne = frequences.unsqueeze(0) #[1, half_dim]

        angles = t_colonne * frequences_ligne #[B, half_dim]

        sinus = torch.sin(angles)  # [B, half_dim]
        cosinus = torch.cos(angles)  # [B, half_dim]

        return torch.cat([sinus, cosinus], dim=1)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.res(x)

class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Encodeur
        self.b1 = Block(in_channels, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.b2 = Block(base_ch * 2, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.mid = Block(base_ch * 4, base_ch * 4, time_dim)
        # Décodeur
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.b_up2 = Block(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.b_up1 = Block(base_ch * 2, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h1 = self.b1(x, t_emb)
        h2 = self.b2(self.down1(h1), t_emb)
        h_mid = self.mid(self.down2(h2), t_emb)
        h_up2 = self.b_up2(torch.cat([self.up2(h_mid), h2], dim=1), t_emb)
        h_up1 = self.b_up1(torch.cat([self.up1(h_up2), h1], dim=1), t_emb)
        return self.out(h_up1)
