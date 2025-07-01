import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        h = self.conv1(x)
        h += time_emb.unsqueeze(-1).unsqueeze(-1)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        return self.downsample(h)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.upconv(x)
        h = torch.cat([x, x], dim=1)
        h = self.conv1(h)
        h += time_emb.unsqueeze(-1).unsqueeze(-1)
        h = F.relu(h)
        h = self.conv2(h)
        return h


class DDPM(nn.Module):
    def __init__(self, image_channels=3, down_channels=(64, 128, 256), up_channels=(256, 128, 64), n_time_embd=128, num_classes=None):
        super().__init__()
        time_embd = n_time_embd
        self.init_conv = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(down_channels[i], down_channels[i+1], time_embd)
            for i in range(len(down_channels)-1)
        ])

        # Middle block
        self.middle_conv = nn.Sequential(
            nn.Conv2d(down_channels[-1], up_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(up_channels[0], up_channels[0], kernel_size=3, padding=1)
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(up_channels[i], up_channels[i+1], time_embd)
            for i in range(len(up_channels)-1)
        ])

        self.final_conv = nn.Conv2d(up_channels[-1], image_channels, kernel_size=3, padding=1)

        # Optional class conditioning
        if num_classes:
            self.label_emb = nn.Embedding(num_classes, time_embd)

    def forward(self, x, t, y=None):
        """
        Args:
            x: [B x C x H x W] noisy input
            t: [B] diffusion step
            y: [B] optional labels for class-conditioned generation
        Returns:
            pred_noise: predicted noise at step t
        """
        x = self.init_conv(x)
        t = t.to(x.device)

        # Class embedding
        if hasattr(self, 'label_emb') and y is not None:
            t = t + self.label_emb(y)

        # Process through U-Net
        residual_inputs = []
        for down_block in self.down_blocks:
            x = down_block(x, t)
            residual_inputs.append(x)

        x = self.middle_conv(x)

        for up_block in self.up_blocks:
            x = torch.cat([x, residual_inputs.pop()], dim=1)
            x = up_block(x, t)

        return self.final_conv(x)