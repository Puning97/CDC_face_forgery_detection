import timm
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class Feature_Transformer_img(nn.Module):
    def __init__(self, *, channels, num_classes,patch_size, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__(),
        self.premodel=timm.create_model('efficientnet_b3a',pretrained=True,num_classes=0,global_pool='')
        self.classes=num_classes
        #num_patches=int(channels)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w-> b c (h w)', c=channels,h=patch_size, w=patch_size),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, int(channels) + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes)
        )

    def forward(self, img):
        #x = self.to_patch_embedding(img)
        feature=self.premodel(img)
        x=self.to_patch_embedding(feature)
        #print('rearrange:',x.shape)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        #print('cls_token:',cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        #print('catcls:',x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        #print('pos_embed:',x.shape)
        x = self.dropout(x)
        #print('drop:',x.shape)
        x = self.transformer(x)
        #print('aftertrans:',x.shape)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #print('mean:',x.shape)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        weight = torch.zeros(b, 1).cuda()
        for i in range(b):
            weight[i] = torch.abs(x[i][0] - x[i][1])
        # print('latent:',x.shape)
        return weight, x
def Feature_Trans_img():
    model=Feature_Transformer_img(channels=1536,num_classes=2,patch_size=7,dim=49,depth=1,heads=6,mlp_dim=1024)
    for k,v in model.named_parameters():
        v.requires_grad=True
    return model