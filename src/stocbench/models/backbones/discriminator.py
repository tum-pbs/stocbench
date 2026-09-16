import copy

import torch.nn as nn

from .unet import get_timestep_embedding


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ADDDiscriminator(nn.Module):
    def __init__(self, original_unet: nn.Module, hidden_dim=128, freeze_backbone=True):
        super().__init__()

        # 1. Setup Backbone (Encoder)
        # Use deepcopy to ensure we don't mutate the generator if it's in memory
        self.backbone = copy.deepcopy(original_unet)

        # Extract components
        self.embed = self.backbone.embed
        self.in_conv = self.backbone.in_conv
        self.downsamples = self.backbone.downsamples
        self.middle = self.backbone.middle

        self.levels = self.backbone.levels
        self.num_res_blocks = self.backbone.num_res_blocks
        self.hid_channels = self.backbone.hid_channels

        # 2. Freeze Backbone (Standard ADD practice)
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

        # 3. Create Multi-Scale Heads
        # We attach a head after the input conv, and after every downsample level
        self.heads = nn.ModuleDict()

        # Head for initial input features
        self.heads["in_conv"] = MLPHead(self.hid_channels, hidden_dim)

        # Heads for downsampling levels
        for i in range(self.levels):
            # Calculate channels for this level based on multipliers
            # Note: You might need to adjust this depending on your specific UNet config implementation
            mult = self.backbone.ch_multipliers[i]
            ch = self.hid_channels * mult
            self.heads[f"level_{i}"] = MLPHead(ch, hidden_dim)

        # Head for the bottleneck (middle)
        # Usually the last multiplier defines the middle channels
        mid_mult = self.backbone.ch_multipliers[-1]
        self.heads["middle"] = MLPHead(self.hid_channels * mid_mult, hidden_dim)

    def forward(self, x, timestep):
        """
        Returns a dictionary of logits from different scales.
        Aggregating them (sum or mean) happens in the loss function.
        """
        # Dictionary to store outputs from all heads
        outputs = {}

        t = timestep.to(x.device)
        t_emb = get_timestep_embedding(t, self.hid_channels)
        t_emb = self.embed(t_emb)

        # --- Forward Pass ---

        # 1. Input Conv
        h = self.in_conv(x)
        outputs["in_conv"] = self.heads["in_conv"](h)

        # 2. Downsampling Levels
        for i in range(self.levels):
            downsample_block = self.downsamples[f"level_{i}"]

            for layer in downsample_block:
                # Robust checking if layer accepts time embedding
                # (Alternative: check based on class type if available)
                try:
                    h = layer(h, t_emb=t_emb)
                except TypeError:
                    h = layer(h)

            # Tap the feature map after the block finishes
            outputs[f"level_{i}"] = self.heads[f"level_{i}"](h)

        h = self.middle(h, t_emb=t_emb)
        outputs["middle"] = self.heads["middle"](h)

        return outputs
