## AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation
## Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2403.14614


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from models.attribute_adapters import SerialDecoderAttrAdjustmentStage
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out
    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')
        
        high_feature, low_feature = self.fft(x) 

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low

# ############## Attribute Models ###################

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class PhotometricColorResidualBlock(nn.Module):
    """
    Photometric-Color residual plug-in for AttributeIR.

    Designed for residual restoration backbones:
        x_out = x + delta_x

    Main idea:
        - Brightness -> low-frequency illumination residual
        - Contrast   -> local contrast expansion residual
        - Color      -> channel-wise chromatic calibration residual

    Inputs:
        x        : [B, C, H, W]    feature to be modulated
        guide_img: [B, 3, H0, W0]  degraded input image (or current RGB guide)
        context  : [B, token_in_dim] or dict
                   default tensor format:
                       context[:, :ctx_token_dim]      -> shared semantic token
                       context[:, ctx_token_dim:]      -> attr deficiencies (length = attr_score_dim)
        steps    : optional, reserved for future scheduling

    Returns:
        out  : [B, C, H, W]
        regs : dict, can be used for regularization / visualization
    """

    def __init__(
        self,
        channels,
        token_in_dim=4096 + 8,
        d_token=256,
        attr_score_dim=8,
        brightness_idx=0,
        contrast_idx=1,
        color_idx=2,
        hidden_channels=None,
        mlp_hidden=128,
        lowfreq_kernel=9,
        local_kernel=7,
        use_guide=True,
        bias=False,
    ):
        super().__init__()

        assert token_in_dim > attr_score_dim
        assert lowfreq_kernel % 2 == 1
        assert local_kernel % 2 == 1

        self.channels = channels
        self.token_in_dim = token_in_dim
        self.attr_score_dim = attr_score_dim
        self.ctx_token_dim = token_in_dim - attr_score_dim

        self.brightness_idx = brightness_idx
        self.contrast_idx = contrast_idx
        self.color_idx = color_idx

        self.lowfreq_kernel = lowfreq_kernel
        self.local_kernel = local_kernel
        self.use_guide = use_guide
        self.eps = 1e-6

        if hidden_channels is None:
            hidden_channels = max(channels // 4, 16)
        self.hidden_channels = hidden_channels

        # ------------------------------------------------------------
        # feature compression
        # ------------------------------------------------------------
        self.feat_reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=bias)

        # guide -> luminance proxy
        # if use_guide=True, we fuse image luminance with feature representation
        guide_in_ch = 1 if use_guide else 0
        illum_in_ch = hidden_channels + guide_in_ch

        self.illum_dw = nn.Conv2d(
            illum_in_ch, illum_in_ch,
            kernel_size=3, padding=1,
            groups=illum_in_ch, bias=bias
        )
        self.illum_pw = nn.Conv2d(illum_in_ch, 1, kernel_size=1, bias=bias)

        # 3-channel chromatic proxy
        self.color_head = nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=bias)

        # ------------------------------------------------------------
        # context projections
        # shared token -> attribute-specific token
        # ------------------------------------------------------------
        self.token_proj_b = nn.Linear(self.ctx_token_dim, d_token, bias=True)
        self.token_proj_c = nn.Linear(self.ctx_token_dim, d_token, bias=True)
        self.token_proj_col = nn.Linear(self.ctx_token_dim, d_token, bias=True)

        # ------------------------------------------------------------
        # conditioning MLPs
        # ------------------------------------------------------------
        cond_feat_dim = hidden_channels + d_token + 1
        self.mlp_b = TinyMLP(cond_feat_dim, mlp_hidden, 1)
        self.mlp_c = TinyMLP(cond_feat_dim, mlp_hidden, 1)

        cond_col_dim = 3 + d_token + 1
        self.mlp_col = TinyMLP(cond_col_dim, mlp_hidden, 6)  # gamma(3) + beta(3)

        # ------------------------------------------------------------
        # residual projection
        # ------------------------------------------------------------
        self.project = nn.Conv2d(3, channels, kernel_size=1, bias=bias)

        # small residual scale for stability in residual backbone
        self.res_scale = nn.Parameter(torch.tensor(0.1))

        # zero-init last projection -> near-identity at start
        nn.init.zeros_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    @staticmethod
    def _to_2d_scalar(x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if x.dim() != 2 or x.size(1) != 1:
            raise ValueError(f"Scalar deficiency must be [B] or [B,1], got {tuple(x.shape)}")
        return x

    def _local_mean(self, x, k):
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def _local_std(self, x, k):
        mean = self._local_mean(x, k)
        mean_sq = self._local_mean(x * x, k)
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        return torch.sqrt(var + self.eps)

    def _rgb_to_luma(self, img):
        """
        img: [B,3,H,W]
        """
        if img.size(1) != 3:
            # fallback
            return img.mean(dim=1, keepdim=True)

        r = img[:, 0:1]
        g = img[:, 1:2]
        b = img[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y

    def _parse_context(self, context, B, device, dtype):
        """
        Supports:
            1) Tensor [B, token_in_dim]
            2) Dict with keys:
               - "shared_token"
               - "attr_scores"
        """
        if context is None:
            shared_token = torch.zeros(B, self.ctx_token_dim, device=device, dtype=dtype)
            attr_scores = torch.zeros(B, self.attr_score_dim, device=device, dtype=dtype)
            return shared_token, attr_scores

        if isinstance(context, dict):
            if "shared_token" not in context or "attr_scores" not in context:
                raise ValueError("Dict context must contain 'shared_token' and 'attr_scores'")
            shared_token = context["shared_token"]
            attr_scores = context["attr_scores"]
            return shared_token, attr_scores

        if torch.is_tensor(context):
            if context.dim() != 2 or context.size(0) != B or context.size(1) != self.token_in_dim:
                raise ValueError(
                    f"context tensor must be [B,{self.token_in_dim}], got {tuple(context.shape)}"
                )
            shared_token = context[:, :self.ctx_token_dim]
            attr_scores = context[:, self.ctx_token_dim:]
            return shared_token, attr_scores

        raise TypeError("context must be None, Tensor, or Dict")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, guide_img, context=None, steps=0):
        """
        x        : [B,C,H,W]
        guide_img: [B,3,H0,W0]
        """
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {C}")

        device = x.device
        dtype = x.dtype

        shared_token, attr_scores = self._parse_context(context, B, device, dtype)

        if attr_scores.size(1) <= max(self.brightness_idx, self.contrast_idx, self.color_idx):
            raise ValueError(
                f"attr_scores dim={attr_scores.size(1)} is too small for "
                f"indices {(self.brightness_idx, self.contrast_idx, self.color_idx)}"
            )

        # ------------------------------------------------------------
        # attribute tokens / deficiencies
        # ------------------------------------------------------------
        z_b = self.token_proj_b(shared_token)      # [B, d_token]
        z_c = self.token_proj_c(shared_token)      # [B, d_token]
        z_col = self.token_proj_col(shared_token)  # [B, d_token]

        d_b = attr_scores[:, self.brightness_idx:self.brightness_idx+1].clamp(0.0, 1.0)
        d_c = attr_scores[:, self.contrast_idx:self.contrast_idx+1].clamp(0.0, 1.0)
        d_col = attr_scores[:, self.color_idx:self.color_idx+1].clamp(0.0, 1.0)

        # ------------------------------------------------------------
        # feature compression
        # ------------------------------------------------------------
        f = self.feat_reduce(x)  # [B, hidden, H, W]

        # ------------------------------------------------------------
        # guide image -> luma proxy
        # ------------------------------------------------------------
        if guide_img is None:
            if self.use_guide:
                raise ValueError("guide_img cannot be None when use_guide=True")
            illum_input = f
            guide_y = None
        else:
            if guide_img.dim() != 4:
                raise ValueError(f"guide_img must be [B,3,H,W], got {tuple(guide_img.shape)}")

            guide_small = F.interpolate(
                guide_img, size=(H, W), mode="bilinear", align_corners=False
            )
            guide_y = self._rgb_to_luma(guide_small)  # [B,1,H,W]

            if self.use_guide:
                illum_input = torch.cat([f, guide_y], dim=1)
            else:
                illum_input = f

        # ------------------------------------------------------------
        # soft illumination proxy
        # ------------------------------------------------------------
        illum_logits = self.illum_pw(self.illum_dw(illum_input))
        L = torch.sigmoid(illum_logits)  # [B,1,H,W]

        L_lf = self._local_mean(L, self.lowfreq_kernel)  # low-frequency illum
        local_mean = self._local_mean(L, self.local_kernel)
        local_std = self._local_std(L, self.local_kernel)
        L_ct = (L - local_mean) / (local_std + self.eps)  # contrast component

        # ------------------------------------------------------------
        # brightness / contrast residual control
        # ------------------------------------------------------------
        gap_f = F.adaptive_avg_pool2d(f, 1).flatten(1)  # [B, hidden]

        cond_b = torch.cat([gap_f, z_b, d_b], dim=1)
        cond_c = torch.cat([gap_f, z_c, d_c], dim=1)

        alpha_b = self.mlp_b(cond_b).view(B, 1, 1, 1)
        alpha_c = self.mlp_c(cond_c).view(B, 1, 1, 1)

        # brightness residual map + contrast residual map
        delta_b = alpha_b * d_b.view(B, 1, 1, 1) * L_lf
        delta_c = alpha_c * d_c.view(B, 1, 1, 1) * L_ct

        # photometric map in residual-compatible form
        # x' ~ x + ((M-1) * x), so here we learn M around 1
        M_pho = 1.0 + delta_b + delta_c
        M_pho = torch.clamp(M_pho, min=0.5, max=1.5)

        # ------------------------------------------------------------
        # color residual control
        # ------------------------------------------------------------
        P = self.color_head(f)  # [B,3,H,W]
        gap_p = F.adaptive_avg_pool2d(P, 1).flatten(1)  # [B,3]

        cond_col = torch.cat([gap_p, z_col, d_col], dim=1)
        gamma_beta = self.mlp_col(cond_col)

        gamma = gamma_beta[:, :3].view(B, 3, 1, 1)
        beta = gamma_beta[:, 3:].view(B, 3, 1, 1)

        # stronger color correction in darker regions
        D = 1.0 - L_lf

        # residual-style photometric + color correction
        # P_prime - P is the real correction residual in proxy space
        P_prime = M_pho * P + D * (gamma * P + beta)

        # back to feature residual
        delta_feat = self.project(P_prime - P)
        out = x + self.res_scale * delta_feat

        regs = {
            "illumination": L,
            "illum_lowfreq": L_lf,
            "illum_contrast": L_ct,
            "delta_b": delta_b,
            "delta_c": delta_c,
            "photometric_map": M_pho,
            "chromatic_proxy": P,
            "chromatic_prime": P_prime,
            "dark_weight": D,
            "gamma": gamma,
            "beta": beta,
            "alpha_b": alpha_b,
            "alpha_c": alpha_c,
            "d_b": d_b,
            "d_c": d_c,
            "d_col": d_col,
            "steps": torch.tensor(float(steps), device=device, dtype=dtype),
        }

        return out, regs

##########################################################################
##---------- AttributeIR with serial decoder adjustment stages -----------------------

class AttributeIR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=True,
        token_in_dim=4096 + 8,
        attr_score_dim=8,
        d_token=256,
        d_ctrl=64,
        hidden_ratio=0.5,
    ):
        super(AttributeIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim*2**1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim*2**2))

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        hidden_l3 = max(int(dim * 4 * hidden_ratio), 24)
        hidden_l2 = max(int(dim * 2 * hidden_ratio), 24)
        hidden_l1 = max(int(dim * 2 * hidden_ratio), 24)

        if self.decoder:
            self.attr_stage_l3 = SerialDecoderAttrAdjustmentStage(
                channels=int(dim*2**2),
                token_in_dim=token_in_dim,
                attr_score_dim=attr_score_dim,
                d_token=d_token,
                d_ctrl=d_ctrl,
                hidden_channels=hidden_l3,
                bias=bias,
                stage_id=0,
            )
            self.attr_stage_l2 = SerialDecoderAttrAdjustmentStage(
                channels=int(dim*2**1),
                token_in_dim=token_in_dim,
                attr_score_dim=attr_score_dim,
                d_token=d_token,
                d_ctrl=d_ctrl,
                hidden_channels=hidden_l2,
                bias=bias,
                stage_id=1,
            )
            self.attr_stage_l1 = SerialDecoderAttrAdjustmentStage(
                channels=int(dim*2**1),
                token_in_dim=token_in_dim,
                attr_score_dim=attr_score_dim,
                d_token=d_token,
                d_ctrl=d_ctrl,
                hidden_channels=hidden_l1,
                bias=bias,
                stage_id=2,
            )

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, context=None, clean_context=None, clean_guide_img=None, steps=0):
        regs_l3 = regs_l2 = regs_l1 = None

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            out_dec_level3, regs_l3 = self.attr_stage_l3(
                out_dec_level3,
                guide_img=inp_img,
                context=context,
                clean_context=clean_context,
                clean_guide_img=clean_guide_img,
                steps=steps,
            )

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2, regs_l2 = self.attr_stage_l2(
                out_dec_level2,
                guide_img=inp_img,
                context=context,
                clean_context=clean_context,
                clean_guide_img=clean_guide_img,
                steps=steps,
            )

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.decoder:
            out_dec_level1, regs_l1 = self.attr_stage_l1(
                out_dec_level1,
                guide_img=inp_img,
                context=context,
                clean_context=clean_context,
                clean_guide_img=clean_guide_img,
                steps=steps,
            )

        out_img = self.output(out_dec_level1) + inp_img
        return out_img, [regs_l3, regs_l2, regs_l1]
