import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Basic UNet blocks
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_ch is None:
            mid_ch = out_ch

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.net(x))
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch),
        )
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

    def forward(self, x, t):
        x = self.block(x)
        emb = self.emb(t)[:, :, None, None]
        return x + emb


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2),
        )
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        emb = self.emb(t)[:, :, None, None]
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.mha = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.ln1 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(
            nn.LayerNorm(ch),
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        x_ln = self.ln1(x)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn
        x = x + self.ff(x)
        return x.transpose(1, 2).view(B, C, H, W)


# ============================================================
# UNet with Concept Vector Injection (STRICT MINIMUM)
# ============================================================

class UNet(nn.Module):
    def __init__(
        self,
        c_in=3,
        c_out=3,
        time_dim=256,
        concept_dim=512,
        concept_scale=1.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.concept_dim = concept_dim
        self.concept_scale = concept_scale

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, time_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256, time_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, time_dim)
        self.sa3 = SelfAttention(256)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, time_dim)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64, time_dim)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64, time_dim)
        self.sa6 = SelfAttention(64)

        self.outc = nn.Conv2d(64, c_out, 1)

    def pos_encoding(self, t, dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=t.device) / dim))
        sin = torch.sin(t * inv_freq)
        cos = torch.cos(t * inv_freq)
        return torch.cat([sin, cos], dim=-1)

    def forward(self, x, t, concept_vector=None):
        B = x.size(0)

        t = t.unsqueeze(-1).float()
        t_emb = self.pos_encoding(t, self.time_dim)

        # -------- Encoder --------
        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t_emb))
        x3 = self.sa2(self.down2(x2, t_emb))
        x4 = self.sa3(self.down3(x3, t_emb))

        # -------- Bottleneck --------
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        if concept_vector is None:
            concept_vector = torch.zeros(B, self.concept_dim, device=x.device)

        x4 = x4 + self.concept_scale * concept_vector[:, :, None, None]

        x4 = self.bot3(x4)

        # -------- Decoder --------
        x = self.sa4(self.up1(x4, x3, t_emb))
        x = self.sa5(self.up2(x, x2, t_emb))
        x = self.sa6(self.up3(x, x1, t_emb))

        return self.outc(x)


def get_model(args):
    model = UNet(
        c_in=args.image_channels,
        c_out=args.image_channels,
        time_dim=args.time_emb_dim,
        concept_dim=args.concept_dim,
        concept_scale=args.concept_scale,
    )
    return model