import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, *, image_size_h,image_size_w, patch_size_h, patch_size_w , num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        assert image_size_h % patch_size_h == 0 and image_size_w % patch_size_w == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size_h // patch_size_h) * (image_size_w // patch_size_w)
        patch_dim = channels *  patch_size_h*  patch_size_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_h, p2 = patch_size_w),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :]
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, :]

        x = self.to_latent(x)
        return self.mlp_head(x)
