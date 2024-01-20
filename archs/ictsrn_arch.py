# Written by KyleChu
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from archs.utils import Upsampler, MeanShift, Conv2d3x3

# --------------------------------------------------------------------------
# LayerNorm
class LayerNormFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, weight, bias, eps):

		ctx.eps = eps
		N, C, H, W = x.size()
		mu = x.mean(1, keepdim=True)
		var = (x - mu).pow(2).mean(1, keepdim=True)
		y = (x - mu) / (var + eps).sqrt()
		ctx.save_for_backward(y, var, weight)
		y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

		return y
	@staticmethod
	def backward(ctx, grad_output):

		eps = ctx.eps

		N, C, H, W = grad_output.size() 
		y, var, weight = ctx.saved_variables
		g = grad_output * weight.view(1, C, 1, 1)
		mean_g = g.mean(dim=1, keepdim=True)

		mean_gy = (g * y).mean(dim=1, keepdim=True)
		gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

		return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
				dim=0), None

class LayerNorm2d(nn.Module):
	def __init__(self, channels, eps=1e-6):
		super(LayerNorm2d, self).__init__()

		self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
		self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
		self.eps = eps

	def forward(self, x):

		return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# --------------------------------------------------------------------------
# Parallel Channel Attention
class Channel(nn.Module):
	def __init__(self, channels):
		super(Channel, self).__init__()

		self.maxpool = nn.AdaptiveMaxPool2d(3)

		self.mlp = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, groups=channels) for _ in range(12)])

	def forward(self, x):

		avg_out = self.maxpool(x)

		output = [nn.Sigmoid()(nn.LeakyReLU()(mlp(avg_out))) for mlp in self.mlp]

		add = nn.LeakyReLU()(sum(output))

		out = nn.LeakyReLU()(add * x)

		return out

# --------------------------------------------------------------------------
# Multi Branch Gate Spatial Convolution
class Spatial(nn.Module):
	def __init__(self, channels):
		super(Spatial, self).__init__()

		self.project_in = nn.Conv2d(channels, channels*3, kernel_size=1)

		self.dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3)

		self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

	def forward(self, x):

		x = self.project_in(x)
		x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
		a = nn.LeakyReLU()(x1 * x2)
		b = nn.LeakyReLU()(a * x3)
		out = self.project_out(b)
		out = nn.LeakyReLU()(out)
		return out

# --------------------------------------------------------------------------
# MLP
class MLP(nn.Module):
	def __init__(self, channels):
		super(MLP, self).__init__()

		self.body = nn.Sequential(
				nn.Conv2d(channels, channels, kernel_size=1),
				nn.LeakyReLU(),
				nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, groups=channels)
				)

	def forward(self, x):

		return self.body(x)

# --------------------------------------------------------------------------
# Efficient Transformer Block
class Transformer(nn.Module):
	def __init__(self, channels):
		super(Transformer, self).__init__()

		self.norm1 = LayerNorm2d(channels)
		self.norm2 = LayerNorm2d(channels)

		self.spatial_blocks = Spatial(channels)
		self.channel_blocks = Channel(channels)

		self.mlp = MLP(channels)

	def forward(self, img):

		x = self.norm1(img)

		x_1 = self.spatial_blocks(x)
		x_2 = nn.LeakyReLU()(x_1)
		x_3 = self.channel_blocks(x_2)
		x_4 = nn.LeakyReLU()(x_3)
		y = x_4 + img

		y_1 = self.norm2(y)

		y_2 = self.mlp(y_1)

		out = y_2 + y

		return out

# --------------------------------------------------------------------------
# Combination CNN and Transforemr
class CTBlock(nn.Module):
	def __init__(self, channels):
		super(CTBlock, self).__init__()

		self.ctblock = nn.Sequential(
				MLP(channels),
				Transformer(channels),
				MLP(channels),
				Transformer(channels),
				MLP(channels),
				Transformer(channels)
				)

	def forward(self, x):

		out = self.ctblock(x) + x

		return out

# --------------------------------------------------------------------------
# Architecture
@ARCH_REGISTRY.register()
class ict(nn.Module):
	def __init__(self, upscale, img_channel=3, channel=64, num_in_ch=3, num_out_ch=3, task="lsr"):

		super(ict, self).__init__()

		self.upscale =  upscale

		self.sub_mean = MeanShift(255, sign=-1, data_type='DF2K')
		self.add_mean = MeanShift(255, sign=1, data_type='DF2K')
		# head
		self.head = Conv2d3x3(img_channel, channel)
		# body
		self.body = nn.Sequential(*[CTBlock(channel) for _ in range(5)])
		# tail
		self.tail = Upsampler(upscale=upscale, in_channels=channel, out_channels=img_channel, upsample_mode=task)

	def forward(self, inp):

		x_sub = self.sub_mean(inp)
		# head
		head_x = self.head(x_sub)
		# body
		body_x = self.body(head_x)
		body_x = body_x + head_x
		# tail
		tail_x = self.tail(body_x)

		out = self.add_mean(tail_x)

		return out
