# Implementation of SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Modified by Yu Tao based on code originally written by Ze Liu and Jingyun Liang for the SwinIR network architecture, https://github.com/JingyunLiang/SwinIR

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    """Multilayer perceptron with two linear layers and an activation.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of features in the hidden layer. Defaults to in_features.
        out_features (int, optional): Number of output features. Defaults to in_features.
        act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
        drop (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)  # First linear layer
        self.act = act_layer()                              # Activation function
        self.fc2 = nn.Linear(hidden_features, out_features) # Second linear layer
        self.drop = nn.Dropout(drop)                        # Dropout layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """Split input tensor into non-overlapping windows.

    Args:
        x (Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): Size of the window.

    Returns:
        Tensor: Windows of shape (num_windows * B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C
    )
    return windows

def window_reverse(windows, window_size, H, W):
    """Reconstruct tensor from windows.

    Args:
        windows (Tensor): Windows of shape (num_windows * B, window_size, window_size, C).
        window_size (int): Size of the window.
        H (int): Original height of the image.
        W (int): Original width of the image.

    Returns:
        Tensor: Reconstructed tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        B, H, W, -1
    )
    return x

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Supports both shifted and non-shifted windows.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, value. Defaults to True.
        qk_scale (float, optional): Scaling factor for query and key. Defaults to None.
        attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout probability for output. Defaults to 0.0.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim                                  # Input dimension
        self.window_size = window_size                  # Window size (Wh, Ww)
        self.num_heads = num_heads                      # Number of attention heads
        head_dim = dim // num_heads                     # Dimension per head
        self.scale = qk_scale or head_dim ** -0.5       # Query scaling factor

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads
            )
        )

        # Compute relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                   # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += window_size[0] - 1                             # Shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                          # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        # Layers for query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)    # Dropout for attention weights
        self.proj = nn.Linear(dim, dim)           # Output projection layer
        self.proj_drop = nn.Dropout(proj_drop)    # Dropout after output projection

        trunc_normal_(self.relative_position_bias_table, std=.02)  # Initialize bias table
        self.softmax = nn.Softmax(dim=-1)                          # Softmax for attention scores

    def forward(self, x, mask=None):
        """Forward pass for window-based self-attention.

        Args:
            x (Tensor): Input features of shape (num_windows * B, N, C).
            mask (Tensor, optional): Attention mask of shape (num_windows, N, N). Defaults to None.

        Returns:
            Tensor: Output features after self-attention.
        """
        B_, N, C = x.shape

        # Compute query, key, value projections
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale                            # Scale query
        attn = (q @ k.transpose(-2, -1))              # Compute attention scores

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # Apply attention mask
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)                   # Apply dropout

        # Compute output features
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)                              # Output projection
        x = self.proj_drop(x)                         # Apply dropout
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        """Calculate FLOPs for the module.

        Args:
            N (int): Number of tokens.

        Returns:
            int: FLOPs for the attention module.
        """
        flops = 0
        flops += N * self.dim * 3 * self.dim             # qkv projection
        flops += self.num_heads * N * (self.dim // self.num_heads) * N  # attention
        flops += self.num_heads * N * N * (self.dim // self.num_heads)  # attention
        flops += N * self.dim * self.dim                 # output projection
        return flops

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional shift for self-attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        num_heads (int): Number of attention heads.
        window_size (int): Window size for self-attention.
        shift_size (int): Shift size for shifted self-attention.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        qkv_bias (bool, optional): If True, adds bias to qkv projections. Defaults to True.
        qk_scale (float, optional): Scaling factor for query and key. Defaults to None.
        drop (float, optional): Dropout probability. Defaults to 0.0.
        attn_drop (float, optional): Attention dropout probability. Defaults to 0.0.
        drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim                                    # Input dimension
        self.input_resolution = input_resolution          # Input resolution
        self.num_heads = num_heads                        # Number of attention heads
        self.window_size = window_size                    # Window size
        self.shift_size = shift_size                      # Shift size
        self.mlp_ratio = mlp_ratio                        # MLP ratio

        if min(self.input_resolution) <= self.window_size:
            # Adjust window size if input resolution is small
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)                      # Layer normalization
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )                                                 # Self-attention module

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # Stochastic depth
        self.norm2 = norm_layer(dim)                      # Layer normalization
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )                                                 # MLP module

        if self.shift_size > 0:
            # Calculate attention mask for shifted windows
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)      # Register attention mask as buffer

    def calculate_mask(self, x_size):
        """Calculate attention mask for SW-MSA.

        Args:
            x_size (tuple[int]): Input resolution.

        Returns:
            Tensor: Attention mask.
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))                # (1, H, W, 1)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)            # (num_windows, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        """Forward pass for Swin Transformer Block.

        Args:
            x (Tensor): Input features of shape (B, H*W, C).
            x_size (tuple[int]): Spatial resolution of input (H, W).

        Returns:
            Tensor: Output features.
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size."

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows * B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (num_windows * B, N, C)

        # Compute attention
        if self.input_resolution == x_size:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # (num_windows * B, N, C)
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # Merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W
        )  # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN with residual connection
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")

    def flops(self):
        """Compute FLOPs for Swin Transformer Block.

        Returns:
            int: FLOPs for the block.
        """
        H, W = self.input_resolution
        flops = 0
        flops += self.dim * H * W                           # LayerNorm
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(
            self.window_size * self.window_size
        )                                                   # Attention
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio  # MLP
        flops += self.dim * H * W                           # LayerNorm
        return flops

class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling.

    Args:
        input_resolution (tuple[int]): Input resolution (H, W).
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution      # Input resolution
        self.dim = dim                                # Input channels
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # Linear reduction
        self.norm = norm_layer(4 * dim)               # Normalization layer

    def forward(self, x):
        """Forward pass for Patch Merging.

        Args:
            x (Tensor): Input features of shape (B, H*W, C).

        Returns:
            Tensor: Output features after downsampling.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size."
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even."

        x = x.view(B, H, W, C)

        # Split into four patches
        x0 = x[:, 0::2, 0::2, :]      # Top-left
        x1 = x[:, 1::2, 0::2, :]      # Bottom-left
        x2 = x[:, 0::2, 1::2, :]      # Top-right
        x3 = x[:, 1::2, 1::2, :]      # Bottom-right

        x = torch.cat([x0, x1, x2, x3], -1)           # Concatenate along channel dimension
        x = x.view(B, -1, 4 * C)                      # (B, H/2*W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)                         # Linear reduction
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """Compute FLOPs for Patch Merging.

        Returns:
            int: FLOPs for the module.
        """
        H, W = self.input_resolution
        flops = H * W * self.dim                      # Normalization
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim  # Linear reduction
        return flops

class BasicLayer(nn.Module):
    """Basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        depth (int): Number of Swin Transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        qkv_bias (bool, optional): If True, adds bias to qkv projections. Defaults to True.
        qk_scale (float, optional): Scaling factor for query and key. Defaults to None.
        drop (float, optional): Dropout probability. Defaults to 0.0.
        attn_drop (float, optional): Attention dropout probability. Defaults to 0.0.
        drop_path (float | list[float], optional): Stochastic depth rate. Defaults to 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        downsample (nn.Module | None, optional): Downsampling layer at the end of the layer. Defaults to None.
        use_checkpoint (bool): If True, uses checkpointing to save memory. Defaults to False.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super(BasicLayer, self).__init__()
        self.dim = dim                                      # Input dimension
        self.input_resolution = input_resolution            # Input resolution
        self.depth = depth                                  # Number of blocks
        self.use_checkpoint = use_checkpoint                # Use gradient checkpointing

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Patch merging layer
        self.downsample = downsample(
            input_resolution, dim=dim, norm_layer=norm_layer
        ) if downsample else None

    def forward(self, x, x_size):
        """Forward pass for BasicLayer.

        Args:
            x (Tensor): Input features.
            x_size (tuple[int]): Spatial resolution (H, W).

        Returns:
            Tensor: Output features.
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"depth={self.depth}")

    def flops(self):
        """Compute FLOPs for BasicLayer.

        Returns:
            int: FLOPs for the layer.
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample:
            flops += self.downsample.flops()
        return flops

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB) with residual connection.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        depth (int): Number of Swin Transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        qkv_bias (bool, optional): If True, adds bias to qkv projections. Defaults to True.
        qk_scale (float, optional): Scaling factor for query and key. Defaults to None.
        drop (float, optional): Dropout probability. Defaults to 0.0.
        attn_drop (float, optional): Attention dropout probability. Defaults to 0.0.
        drop_path (float | list[float], optional): Stochastic depth rate. Defaults to 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        downsample (nn.Module | None, optional): Downsampling layer at the end of the layer. Defaults to None.
        use_checkpoint (bool): If True, uses checkpointing to save memory. Defaults to False.
        img_size (int): Input image size.
        patch_size (int): Patch size.
        resi_connection (str): Type of residual connection ('1conv' or '3conv').
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, img_size=224,
                 patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim                                  # Input dimension
        self.input_resolution = input_resolution        # Input resolution

        # Residual group of Swin Transformer blocks
        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)  # Convolutional layer
        elif resi_connection == '3conv':
            # Three convolutional layers with LeakyReLU activations
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )

        # Embedding and unembedding layers
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim, norm_layer=None
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim, norm_layer=None
        )

    def forward(self, x, x_size):
        """Forward pass for RSTB.

        Args:
            x (Tensor): Input features.
            x_size (tuple[int]): Spatial resolution (H, W).

        Returns:
            Tensor: Output features after residual connection.
        """
        res = self.residual_group(x, x_size)
        res = self.patch_unembed(res, x_size)
        res = self.conv(res)
        res = self.patch_embed(res)
        x = x + res
        return x

    def flops(self):
        """Compute FLOPs for RSTB.

        Returns:
            int: FLOPs for the block.
        """
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9          # Convolution
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops

class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int): Image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        ]
        self.img_size = img_size                      # Image size
        self.patch_size = patch_size                  # Patch size
        self.patches_resolution = patches_resolution  # Resolution after patching
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans                      # Input channels
        self.embed_dim = embed_dim                    # Embedding dimension
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        """Forward pass for patch embedding.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Flattened and embedded patches.
        """
        x = x.flatten(2).transpose(1, 2)             # (B, num_patches, C)
        if self.norm:
            x = self.norm(x)
        return x

    def flops(self):
        """Compute FLOPs for PatchEmbed.

        Returns:
            int: FLOPs for the module.
        """
        H, W = self.img_size
        flops = 0
        if self.norm:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    """Patch Unembedding to reconstruct image from patches.

    Args:
        img_size (int): Image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super(PatchUnEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        ]
        self.img_size = img_size                      # Image size
        self.patch_size = patch_size                  # Patch size
        self.patches_resolution = patches_resolution  # Resolution after patching
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans                      # Input channels
        self.embed_dim = embed_dim                    # Embedding dimension

    def forward(self, x, x_size):
        """Forward pass for patch unembedding.

        Args:
            x (Tensor): Input tensor.
            x_size (tuple[int]): Spatial resolution (H, W).

        Returns:
            Tensor: Reconstructed image tensor.
        """
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(
            B, self.embed_dim, x_size[0], x_size[1]
        )  # (B, C, H, W)
        return x

    def flops(self):
        """Compute FLOPs for PatchUnEmbed.

        Returns:
            int: FLOPs for the module (zero in this case).
        """
        return 0

class Upsample(nn.Sequential):
    """Upsampling module using PixelShuffle.

    Args:
        scale (int): Upscaling factor.
        num_feat (int): Number of feature channels.
    """
    def __init__(self, scale, num_feat):
        layers = []
        if (scale & (scale - 1)) == 0:   # Is scale a power of 2?
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError("Unsupported scale factor.")
        super(Upsample, self).__init__(*layers)

class UpsampleOneStep(nn.Sequential):
    """Single-step upsampling module for lightweight SR.

    Args:
        scale (int): Upscaling factor.
        num_feat (int): Number of feature channels.
        num_out_ch (int): Number of output channels.
        input_resolution (tuple[int], optional): Input resolution. Defaults to None.
    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        layers = [
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale)
        ]
        super(UpsampleOneStep, self).__init__(*layers)

    def flops(self):
        """Compute FLOPs for UpsampleOneStep.

        Returns:
            int: FLOPs for the module.
        """
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class SwinIR(nn.Module):
    """SwinIR: Image Restoration Using Swin Transformer.

    Args:
        img_size (int | tuple[int]): Input image size. Defaults to 64.
        patch_size (int | tuple[int]): Patch size. Defaults to 1.
        in_chans (int): Number of input image channels. Defaults to 3.
        embed_dim (int): Embedding dimension. Defaults to 96.
        depths (list[int]): Depth of each Swin Transformer layer.
        num_heads (list[int]): Number of attention heads in each layer.
        window_size (int): Window size. Defaults to 7.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.
        qkv_bias (bool): If True, adds bias to qkv projections. Defaults to True.
        qk_scale (float): Scaling factor for query and key. Defaults to None.
        drop_rate (float): Dropout probability. Defaults to 0.0.
        attn_drop_rate (float): Attention dropout probability. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        norm_layer (nn.Module): Normalization layer. Defaults to nn.LayerNorm.
        ape (bool): If True, uses absolute position embedding. Defaults to False.
        patch_norm (bool): If True, adds normalization after patch embedding. Defaults to True.
        use_checkpoint (bool): If True, uses checkpointing to save memory. Defaults to False.
        upscale (int): Upscaling factor. Defaults to 2.
        img_range (float): Image range. Defaults to 1.0.
        upsampler (str): Upsampling method. Choices: '', 'pixelshuffle', 'pixelshuffledirect', 'nearest+conv'.
        resi_connection (str): Residual connection type. Choices: '1conv', '3conv'.
    """
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1.,
                 upsampler='', resi_connection='1conv', **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        ###################################
        # 1. Shallow feature extraction
        ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        ###################################
        # 2. Deep feature extraction
        ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape                                   # Absolute position embedding
        self.patch_norm = patch_norm                     # Apply normalization after patch embedding
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Merge patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)          # Dropout after position embedding

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths)
        )]

        # Build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim, input_resolution=(
                    patches_resolution[0], patches_resolution[1]
                ),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]):sum(depths[:i_layer + 1])
                ],
                norm_layer=norm_layer, downsample=None,
                use_checkpoint=use_checkpoint, img_size=img_size,
                patch_size=patch_size, resi_connection=resi_connection
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # Last convolutional layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(
                embed_dim, embed_dim, 3, 1, 1
            )
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        ###################################
        # 3. High-quality image reconstruction
        ###################################
        if self.upsampler == 'pixelshuffle':
            # For classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # For lightweight SR
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1])
            )
        elif self.upsampler == 'nearest+conv':
            # For real-world SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # For image denoising and compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """No weight decay for certain parameters."""
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """No weight decay keywords."""
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        """Pad image to be a multiple of window size.

        Args:
            x (Tensor): Input image.

        Returns:
            Tensor: Padded image.
        """
        _, _, h, w = x.size()
        mod_pad_h = (
            self.window_size - h % self.window_size
        ) % self.window_size
        mod_pad_w = (
            self.window_size - w % self.window_size
        ) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """Extract features using Swin Transformer layers.

        Args:
            x (Tensor): Input features.

        Returns:
            Tensor: Output features.
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # (B, L, C)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        """Forward pass for SwinIR.

        Args:
            x (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # Classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # Lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x)
            ) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # Real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    F.interpolate(x, scale_factor=2, mode='nearest')
                )
            )
            if self.upscale == 4:
                x = self.lrelu(
                    self.conv_up2(
                        F.interpolate(x, scale_factor=2, mode='nearest')
                    )
                )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # Image denoising and compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(
                self.forward_features(x_first)
            ) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        """Compute FLOPs for SwinIR.

        Returns:
            int: FLOPs for the model.
        """
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9        # Conv_first
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim  # Conv_after_body
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops()
        return flops

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(
        upscale=2, img_size=(height, width), window_size=window_size,
        img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
        num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect'
    )
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)

