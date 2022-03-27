# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
from IPython.core.debugger import set_trace

import torchvision.models as models
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ACD_VGG_BN_256(nn.Module):
    def __init__(self, in_channels, nclass):
        super(ACD_VGG_BN_256, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
        #    nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
#            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
#            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
#            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
#            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
#            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
#            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
            
            nn.Flatten(),
        )

         
        self.gan1 = nn.Sequential(
            nn.Linear(512, 1000),
            nn.LeakyReLU(0.1, True),
            nn.Linear(1000, nclass)
        )





    def forward(self, x):

        fea = self.feature(x[:,:,:,:])
        gan1 = self.gan1(fea)
#        fea = 0
#        
#        
#        fea = self.feature[0:20](x)
#        fea = fea.view(fea.size(0), -1)
#        gan2 = self.gan2(fea)
#        fea = 0
#        
#        
#        fea = self.feature[0:16](x)
#        fea = fea.view(fea.size(0), -1)
#        
#        gan3 = self.gan3(fea)
#        fea = 0
#        
#        
#        fea = self.feature[0:12](x)
#        fea = fea.view(fea.size(0), -1)
#        
#        gan4 = self.gan4(fea)
#        fea = 0
#        
#        
#        fea = self.feature[0:8](x)
#        fea = fea.view(fea.size(0), -1)
#        
#        gan5 = self.gan5(fea)
#        fea = 0
#        
#        
#        fea = self.feature[0:4](x)
#        fea = fea.view(fea.size(0), -1)
#        
#        gan6 = self.gan6(fea)
#        fea = 0
#        
        
        #cls = self.cls(fea)
        return gan1  






###########################
#    Vision Transformer   #
###########################



import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 im_len, 
                 im_ch,
                 n_class, 
                 patch_dim=1, 
                 flatten_dim=64, 
                 t_heads=8, 
                 t_depth=6, 
                 mlp_dim=1024, 
                 dropout=0.1, 
                 t_do=0.1, 
                ):
        super().__init__()
        self.p = patch_dim
        assert im_len % patch_dim == 0
        n_patch = (im_len // patch_dim) ** 2
        self.patch_len = im_ch * patch_dim ** 2
        self.flatten_dim = flatten_dim
        
        self.position = nn.Parameter(torch.empty(1, n_patch+1, flatten_dim))
        torch.nn.init.normal_(self.position, std = .05) # .05, .01, .005, .001
        
        self.lin_proj= nn.Linear(self.patch_len, flatten_dim)
        self.embed = False
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, flatten_dim))
        self.drop = nn.Dropout(dropout)

        self.t = Transformer(flatten_dim, t_depth, t_heads, mlp_dim, t_do)
#         torch.nn.Transformer(
#             d_model=flatten_dim, 
#             nhead=t_heads, 
#             dim_feedforward=mlp_dim, 
#             dropout=t_do
#         )

        self.to_cls_token = nn.Identity()

        self.mlp = nn.Sequential(
            nn.LayerNorm(flatten_dim),
            nn.Linear(flatten_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, n_class),
        )
        
        # self.pixel_mlp = nn.Sequential(
            # nn.Linear(im_ch, im_ch*2), 
            # nn.Tanh(),
        # )

    def add_lin_proj(self, preembed_file, inp=11):
        # self.lin_proj = nn.Linear(self.patch_len, flatten_dim)
        self.lin_proj = nn.Sequential(nn.Linear(inp, 32), nn.Tanh(), nn.Linear(32, 64))
        self.lin_proj.load_state_dict(torch.load(preembed_file))
        self.embed = True
        self.no_grad=False
    
    def allow_embed_grad(self):
        self.no_grad=True
    
    def forward(self, x, t_mask=None):
        # https://stackoverflow.com/questions/41904598/how-to-rearrange-an-array-by-subarray-elegantly-in-numpy
        # Indexing expression contains duplicate dimension
        
        # half = x.shape[-1] // 2
        # center = x[:, :, half, half]
        # center = self.pixel_mlp(center)
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.p, p2 = self.p)
        if self.embed:
            if self.no_grad:
                with torch.no_grad():
                    x = self.lin_proj(x)
            else:
                x = self.lin_proj(x)
        else:
            x = self.lin_proj(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position[:, :(x.shape[1] + 1)]
        x = self.drop(x)

        x = self.t(x, t_mask)

        x = self.to_cls_token(x[:, 0])
        x = self.mlp(x)
        # x = self.mlp(torch.cat([x, center], dim=1))
        
        return x
 