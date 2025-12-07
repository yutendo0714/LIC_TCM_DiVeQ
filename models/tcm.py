from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
from .diveq import DiVeQ, SFDiVeQ


class CategoricalArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.low = 0
        self.high = (1 << precision) - 1
        self.pending_bits = 0
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_filled = 0
        self.mask = (1 << precision) - 1
        self.half = 1 << (precision - 1)
        self.quarter = self.half >> 1
        self.three_quarter = self.quarter * 3

    def _push_bit(self, bit):
        self.current_byte = (self.current_byte << 1) | bit
        self.bits_filled += 1
        if self.bits_filled == 8:
            self.buffer.append(self.current_byte & 0xFF)
            self.current_byte = 0
            self.bits_filled = 0

    def _output_bit_with_pending(self, bit):
        self._push_bit(bit)
        complement = bit ^ 1
        while self.pending_bits > 0:
            self._push_bit(complement)
            self.pending_bits -= 1

    def encode_symbol(self, cdf_row, symbol):
        total = int(cdf_row[-1])
        low_count = int(cdf_row[symbol])
        high_count = int(cdf_row[symbol + 1])
        if high_count <= low_count:
            raise ValueError("Invalid categorical CDF entry.")
        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * high_count // total) - 1
        self.low = self.low + (range_ * low_count // total)

        while True:
            if self.high < self.half:
                self._output_bit_with_pending(0)
            elif self.low >= self.half:
                self._output_bit_with_pending(1)
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarter:
                self.pending_bits += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1

    def flush(self):
        self.pending_bits += 1
        if self.low < self.quarter:
            self._output_bit_with_pending(0)
        else:
            self._output_bit_with_pending(1)

        if self.bits_filled > 0:
            self.buffer.append((self.current_byte << (8 - self.bits_filled)) & 0xFF)
            self.current_byte = 0
            self.bits_filled = 0
        return bytes(self.buffer)


class CategoricalArithmeticDecoder:
    def __init__(self, data, precision=32):
        self.precision = precision
        self.low = 0
        self.high = (1 << precision) - 1
        self.code = 0
        self.data = data
        self.byte_index = 0
        self.bits_remaining = 0
        self.current_byte = 0
        self.mask = (1 << precision) - 1
        self.half = 1 << (precision - 1)
        self.quarter = self.half >> 1
        self.three_quarter = self.quarter * 3

        for _ in range(precision):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self.bits_remaining == 0:
            if self.byte_index < len(self.data):
                self.current_byte = self.data[self.byte_index]
                self.byte_index += 1
            else:
                self.current_byte = 0
            self.bits_remaining = 8
        bit = (self.current_byte >> 7) & 1
        self.current_byte = (self.current_byte << 1) & 0xFF
        self.bits_remaining -= 1
        return bit

    def decode_symbol(self, cdf_row):
        total = int(cdf_row[-1])
        range_ = self.high - self.low + 1
        scaled_value = ((self.code - self.low + 1) * total - 1) // range_
        symbol = int(np.searchsorted(cdf_row, scaled_value + 1, side="left") - 1)
        symbol = max(0, min(symbol, len(cdf_row) - 2))
        low_count = int(cdf_row[symbol])
        high_count = int(cdf_row[symbol + 1])
        self.high = self.low + (range_ * high_count // total) - 1
        self.low = self.low + (range_ * low_count // total)

        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarter:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1
            self.code = ((self.code << 1) & self.mask) | self._read_bit()
        return symbol


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x

class TCM(CompressionModel):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128,  M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [conv3x3(2*N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 2*N, 2)] + \
            self.ha_down1
        )

        self.hs_up1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up1
        )

        self.hs_up2 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]


        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up2
        )

        self.vq_type = kwargs.get("vq_type", "diveq").lower()
        if self.vq_type not in ("diveq", "sfdiveq"):
            raise ValueError(f"Unsupported vq_type: {self.vq_type}")
        self.codebook_size = kwargs.get("codebook_size", 512)
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2")
        self.vq_symbol_size = self.codebook_size if self.vq_type == "diveq" else self.codebook_size - 1
        if self.vq_symbol_size < 1:
            raise ValueError("SF-DiVeQ requires codebook_size >= 2")
        self.vq_sigma_squared = kwargs.get("vq_sigma_squared", 1e-3)
        self.vq_commitment_cost = kwargs.get("vq_commitment_cost", 0.25)
        self.vq_warmup_steps = kwargs.get("vq_warmup_steps", 0)
        self.sf_init_warmup_epochs = kwargs.get("sf_init_warmup_epochs", 2)
        self.sf_sigma_squared = kwargs.get("sf_sigma_squared", 1e-2)
        self.vq_use_codebook_replacement = kwargs.get("vq_use_codebook_replacement", True)
        self.vq_replacement_config = kwargs.get("vq_replacement_config", None)
        self.categorical_coder_precision = kwargs.get("categorical_coder_precision", 32)
        self.cdf_precision = kwargs.get("categorical_cdf_precision", 12)
        self.cdf_total = 1 << self.cdf_precision
        slice_dim = M // self.num_slices
        vq_layers = []
        for _ in range(self.num_slices):
            if self.vq_type == "diveq":
                vq_layers.append(
                    DiVeQ(
                        num_embeddings=self.codebook_size,
                        embedding_dim=slice_dim,
                        sigma_squared=self.vq_sigma_squared,
                        commitment_cost=self.vq_commitment_cost,
                        use_codebook_replacement=self.vq_use_codebook_replacement,
                        replacement_config=self.vq_replacement_config,
                    )
                )
            else:
                vq_layers.append(
                    SFDiVeQ(
                        num_embeddings=self.codebook_size,
                        embedding_dim=slice_dim,
                        sigma_squared=self.sf_sigma_squared,
                        commitment_cost=self.vq_commitment_cost,
                        init_warmup_epochs=self.sf_init_warmup_epochs,
                    )
                )
        self.vq_layers = nn.ModuleList(vq_layers)
        self.register_buffer("vq_step", torch.zeros(1, dtype=torch.long))


        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_logits_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.vq_symbol_size, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)

    def update(self, scale_table=None, force=False):
        del scale_table
        return super().update(force=force)

    def _build_categorical_cdf(self, logits):
        symbol_dim = logits.size(1)
        total = max(int(self.cdf_total), int(symbol_dim) + 1)
        probs = torch.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1).contiguous()
        flat = probs.view(-1, symbol_dim).detach()
        spread = total - symbol_dim
        scaled = flat * spread
        base = torch.floor(scaled)
        counts = (base + 1).long()
        remainder = (total - counts.sum(dim=1)).long()
        fractional = (scaled - base)

        counts = counts.cpu()
        fractional = fractional.cpu()
        remainder = remainder.cpu()

        for idx in range(counts.size(0)):
            r = int(remainder[idx].item())
            if r <= 0:
                continue
            full_cycles, leftover = divmod(r, symbol_dim)
            if full_cycles > 0:
                counts[idx] += full_cycles
            if leftover > 0:
                topk = torch.topk(fractional[idx], k=leftover, sorted=False).indices
                counts[idx, topk] += 1

        cdf = torch.zeros(counts.size(0), symbol_dim + 1, dtype=torch.int64)
        cdf[:, 1:] = torch.cumsum(counts, dim=1)
        cdf[:, -1] = total
        return np.ascontiguousarray(cdf.numpy())

    def _sample_lambda_pairs(self, generator, device, dtype):
        if generator is None:
            raise RuntimeError("Lambda generator is not initialized.")
        values = torch.rand(
            self.codebook_size - 1,
            1,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        return values.to(device=device, dtype=dtype)
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        logits_list = []
        vq_losses = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)

            prob_support = torch.cat([latent_scales] + support_slices, dim=1)
            prob_support = self.atten_scale[slice_index](prob_support)
            logits = self.cc_logits_transforms[slice_index](prob_support)
            logits = logits[:, :, :y_shape[0], :y_shape[1]]
            logits_list.append(logits)

            centered = (y_slice - mu).permute(0, 2, 3, 1)
            vq_module = self.vq_layers[slice_index]
            vq_kwargs = {}
            if isinstance(vq_module, SFDiVeQ):
                skip_quant = self.training and (int(self.vq_step.item()) < self.vq_warmup_steps)
                vq_kwargs["skip_quantization"] = skip_quant
            quantized_residual, vq_loss, indices = vq_module(centered, **vq_kwargs)
            if vq_loss is None:
                vq_losses.append(centered.new_zeros(()))
            else:
                vq_losses.append(vq_loss)
            quantized_residual = quantized_residual.permute(0, 3, 1, 2)
            y_hat_slice = mu + quantized_residual

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            log_probs = F.log_softmax(logits, dim=1)
            gathered_log_probs = torch.gather(log_probs, 1, indices.unsqueeze(1))
            symbol_probs = torch.exp(gathered_log_probs)
            y_likelihood.append(symbol_probs)

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        logits_cat = torch.cat(logits_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        if self.training:
            self.vq_step += 1
        vq_loss_total = torch.stack(vq_losses).mean()

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "logits": logits_cat, "y": y},
            "vq_loss": vq_loss_total,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if x.size(0) != 1:
            raise ValueError("Compression only supports batch size 1.")
        with torch.no_grad():
            y = self.g_a(x)
            y_shape = y.shape[2:]
            z = self.h_a(y)
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

            latent_scales = self.h_scale_s(z_hat)
            latent_means = self.h_mean_s(z_hat)

            y_slices = y.chunk(self.num_slices, 1)
            y_hat_slices = []
            encoder = CategoricalArithmeticEncoder(self.categorical_coder_precision)
            lambda_seed = None
            lambda_gen = None
            if self.vq_type == "sfdiveq":
                lambda_seed = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())
                lambda_gen = torch.Generator(device="cpu")
                lambda_gen.manual_seed(lambda_seed)

            for slice_index, y_slice in enumerate(y_slices):
                support_slices = (
                    y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices]
                )
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                mean_support = self.atten_mean[slice_index](mean_support)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                prob_support = torch.cat([latent_scales] + support_slices, dim=1)
                prob_support = self.atten_scale[slice_index](prob_support)
                logits = self.cc_logits_transforms[slice_index](prob_support)
                logits = logits[:, :, :y_shape[0], :y_shape[1]]

                centered = (y_slice - mu).permute(0, 2, 3, 1)
                lambda_pairs = None
                if self.vq_type == "sfdiveq":
                    lambda_pairs = self._sample_lambda_pairs(lambda_gen, centered.device, centered.dtype)
                quantized_residual, _, indices = self.vq_layers[slice_index](
                    centered, return_loss=False, lambda_pairs=lambda_pairs
                )
                indices_np = indices.view(-1).cpu().numpy()
                cdf = self._build_categorical_cdf(logits)
                cdf_rows = cdf.reshape(-1, cdf.shape[-1])
                for idx_symbol, symbol in enumerate(indices_np):
                    encoder.encode_symbol(cdf_rows[idx_symbol], int(symbol))

                quantized_residual = quantized_residual.permute(0, 3, 1, 2)
                y_hat_slice = mu + quantized_residual
                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

            y_string = encoder.flush()
            y_payload = [y_string]
            if lambda_seed is not None:
                y_payload.append(int(lambda_seed).to_bytes(8, byteorder="little", signed=False))

        return {"strings": [y_payload, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        with torch.no_grad():
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
            latent_scales = self.h_scale_s(z_hat)
            latent_means = self.h_mean_s(z_hat)

            y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
            y_payload = strings[0]
            if not isinstance(y_payload, (list, tuple)) or len(y_payload) == 0:
                raise ValueError("Invalid bitstream payload.")
            decoder = CategoricalArithmeticDecoder(y_payload[0], precision=self.categorical_coder_precision)
            lambda_gen = None
            if self.vq_type == "sfdiveq":
                if len(y_payload) < 2:
                    raise ValueError("Missing lambda seed for SF-DiVeQ stream.")
                lambda_seed = int.from_bytes(y_payload[1], byteorder="little", signed=False)
                lambda_gen = torch.Generator(device="cpu")
                lambda_gen.manual_seed(lambda_seed)

            y_hat_slices = []
            device = latent_means.device
            for slice_index in range(self.num_slices):
                support_slices = (
                    y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices]
                )
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                mean_support = self.atten_mean[slice_index](mean_support)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                prob_support = torch.cat([latent_scales] + support_slices, dim=1)
                prob_support = self.atten_scale[slice_index](prob_support)
                logits = self.cc_logits_transforms[slice_index](prob_support)
                logits = logits[:, :, :y_shape[0], :y_shape[1]]

                cdf = self._build_categorical_cdf(logits)
                cdf_rows = cdf.reshape(-1, cdf.shape[-1])
                num_symbols = cdf_rows.shape[0]
                decoded = np.empty(num_symbols, dtype=np.int64)
                for idx_symbol in range(num_symbols):
                    decoded[idx_symbol] = decoder.decode_symbol(cdf_rows[idx_symbol])
                indices = torch.from_numpy(decoded).to(device).view(1, y_shape[0], y_shape[1])

                codebook = self.vq_layers[slice_index].codebook.weight
                lambda_pairs = None
                if self.vq_type == "sfdiveq":
                    lambda_pairs = self._sample_lambda_pairs(lambda_gen, device, latent_means.dtype)
                indices_flat = indices.view(-1)
                if self.vq_type == "sfdiveq":
                    if lambda_pairs is None:
                        raise RuntimeError("Lambda pairs are required for SF-DiVeQ.")
                    lambda_selected = lambda_pairs[indices_flat].to(device)
                    c_i = codebook[indices_flat]
                    c_i_plus = codebook[indices_flat + 1]
                    quantized_flat = (1.0 - lambda_selected) * c_i + lambda_selected * c_i_plus
                else:
                    quantized_flat = F.embedding(indices_flat, codebook)

                quantized_flat = quantized_flat.view(1, y_shape[0], y_shape[1], -1)
                quantized_residual = quantized_flat.permute(0, 3, 1, 2)
                y_hat_slice = mu + quantized_residual

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

            y_hat = torch.cat(y_hat_slices, dim=1)
            x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
