# Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        patch_size=32,
        dim=1024,
        out_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        channels=3,
        dim_head=64,
        use_positional_embedding=True,
        add_cls_token=True,
        resize_image_to_px=64,
    ):
        super().__init__()
        
        # Store these as instance variables
        self.use_positional_embedding = use_positional_embedding
        self.add_cls_token = add_cls_token
        
        patch_dim = channels * patch_size**2
        self.num_patches = (resize_image_to_px // patch_size) ** 2  # Assuming 224x224 images

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.out_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )
        self.out_layer2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 200),
        )
        if(self.use_positional_embedding):
            if(self.add_cls_token):
                self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
            else:
                self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
            
        if(self.add_cls_token):
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        #print("after patch embedding shape: ", x.shape)
        if(self.add_cls_token):
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            #print("after cls token shape: ", x.shape)
        if(self.use_positional_embedding):
            x = self.positional_embedding + x
            print("after positional embedding shape: ", x.shape)

        x = self.transformer(x)
        #x = self.out_layer(x)
        x_class = self.out_layer2(x[:, 0, :]) #use CLS token for classification
        #print("after out layer shape: ", x_class.shape)
        return x_class
