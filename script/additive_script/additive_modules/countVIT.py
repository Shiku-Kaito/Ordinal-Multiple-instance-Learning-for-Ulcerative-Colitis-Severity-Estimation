import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)

        # attn = F.softmax(attn / 0.1 ,dim=-1) 
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


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


class countVIT(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        device = "cuda:0"
    ):
        super().__init__()
        # self.feature_extractor = resnet18(pretrained=False)
        # self.feature_extractor.fc = nn.Sequential()

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
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.num_classes = num_classes
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.device = device


    def forward(self, x):
        # Batch, Bag, C, w, h = x.shape
        # x = x.reshape(-1, C, w, h)

        # token embedding
        # x = self.feature_extractor(x)
        Batch, Bag, emb_dim = x.shape
        x = x.reshape(-1, 512)
        x = self.fc(x)
        x = x.reshape(Batch, Bag, -1)
        
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1).to('cuda:0')
        # cls_tokens = self.fc(cls_tokens)
        # x = torch.cat((cls_tokens, x), dim=1)

        # transformer
        attn_weights = []
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

        # x = self.norm(x)

        # # classifier
        # x = self.head(x)
        # y_ins = F.softmax(x / 0.1 ,dim=2)
        # count = y_ins.sum(dim=1)      #カウントする
        # count /= count.sum(axis=1, keepdims=True)    #正規化
        # y_bag = F.softmax(count / 1 ,dim=1)     #softmax with temperature
        x = self.norm(x)
        x = x.reshape(-1, 768)
        x = self.head(x)
        # # cls_bag = self.head(x[:, :1, :]).squeeze()
        # cls_bag = self.head(x)
        # cls_bag = F.softmax(cls_bag, dim = 1)

        # y_bag_feat_dict = {'Y_prob' : cls_bag}
        return x #, x.reshape(-1,self.num_classes),  y_bag  # batch, instance, class


if __name__ == "__main__":
    x = torch.randn(1, 16, 3, 224, 224)
    model = countVIT()

    y = model(x)

    print(1)
