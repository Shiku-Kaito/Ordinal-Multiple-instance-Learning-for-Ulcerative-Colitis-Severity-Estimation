import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
import copy

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, num_classes=20, clstoken_mask=True):
        super().__init__()
        # self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.clstoken_mask = clstoken_mask

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        if self.clstoken_mask==1:
            mask_matrix = self.gen_mask(attn)
            attn = torch.mul(attn, mask_matrix)

        # pre_soft_attn = copy.deepcopy(attn)
        
        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights

    def gen_mask(self, attn):
        pre_attn = attn.sum(dim=1)/ self.num_heads
        mask_matrix = torch.ones(pre_attn.shape).to(attn.device)
        # one vs rest token mask
        mask_matrix[:,0,1:3] = 0.0
        mask_matrix[:,1,0],mask_matrix[:,1,2] = 0.0, 0.0
        mask_matrix[:,2,:2] = 0.0 
 
        mask_matrixes = []
        for i in range(mask_matrix.shape[0]):
            mask_matrixes.append(mask_matrix[i].repeat(self.num_heads,1,1))
        mask_matrixes = torch.stack((mask_matrixes))
        return mask_matrixes



class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_classes=20,
        clstoken_mask=True
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_classes=num_classes,
            clstoken_mask=clstoken_mask
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class Selective_Aggregated_Transfomer(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        clstoken_mask=True
    ):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()

        self.fc = nn.Linear(512, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_classes=num_classes,
                    clstoken_mask=clstoken_mask
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 3, embed_dim))

        self.head1 = nn.Linear(embed_dim, num_classes)
        self.head2 = nn.Linear(embed_dim, num_classes)
        self.head3 = nn.Linear(embed_dim, num_classes)

    def forward(self, x, len_list):
        ins_feat = self.feature_extractor(x)
        ins_feat = self.fc(ins_feat)

        ini_idx = 0
        ins_feats = []
        for length in len_list:
            ins_feats.append(ins_feat[ini_idx : ini_idx + length])
            ini_idx += length
    
        x = pad_sequence(ins_feats, batch_first=True, padding_value=0)
        
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # transformer
        attn_weights = []
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights = (weights_i.sum(dim=1)/ 8)
            # pre_soft_attn = (pre_soft_attn.sum(dim=1)/ 8)

        x = self.norm(x)

        x1 = self.head1(x[:, :1, :]).squeeze()
        x2 = self.head2(x[:, 1:2, :]).squeeze()
        x3 = self.head3(x[:, 2:3, :]).squeeze()

        # ins_feats = []
        # for idx, length in enumerate(len_list):
        #     ins_feats.append(x[idx][3 : length+3])

        return {"y_0vs123": x1, "y_01vs23":x2, "y_012vs3":x3, "atten_weight":attn_weights, "bag_feat": x[:,:3,:], "ins_feats": ins_feats, "cls_token_feat": self.cls_token}


