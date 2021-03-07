import sys
import torch
import torch.nn as nn

from option import args
from collections import OrderedDict


# ------helper functions------ #

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type=args.act_type, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!' % norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# ------build blocks------ #
def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, conv, n, act)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, act_type='relu',
                norm_type='bn', pad_type='zero'):
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, deconv, n, act)


# ------build SRB ------ #
class SRB(nn.Module):
    def __init__(self, norm_type):
        super(SRB, self).__init__()
        upscale_factor = args.scale
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8

        self.num_groups = args.num_groups
        num_features = args.num_features
        act_type = args.act_type

        self.compress_in = ConvBlock(num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                            act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(
                ConvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                          act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features, kernel_size=1, stride=1, act_type=act_type,
                              norm_type=norm_type))
                self.downtranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features, kernel_size=1, stride=1, act_type=act_type,
                              norm_type=norm_type))

        self.compress_out = ConvBlock(self.num_groups * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)
        self.last_hidden = None

    def forward(self, f_in):
        # use cuda
        f = torch.zeros(f_in.size()).cuda()
        f.copy_(f_in)

        f = self.compress_in(f)

        lr_features = []
        hr_features = []
        lr_features.append(f)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        g = torch.cat(tuple(lr_features[1:]), 1)
        g = self.compress_out(g)

        return g


# ------build CFB ------ #
class CFB(nn.Module):
    def __init__(self, norm_type):
        super(CFB, self).__init__()
        upscale_factor = args.scale
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8

        self.num_groups = args.num_groups
        num_features = args.num_features
        act_type = args.act_type

        self.compress_in = ConvBlock(3 * num_features, num_features, kernel_size=1, act_type=act_type,
                                     norm_type=norm_type)
        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        self.re_guide = ConvBlock(2 * num_features, num_features, kernel_size=1, act_type=act_type,
                                  norm_type=norm_type)
        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                            act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(
                ConvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                          act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features, kernel_size=1, stride=1, act_type=act_type,
                              norm_type=norm_type))
                self.downtranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features, kernel_size=1, stride=1, act_type=act_type,
                              norm_type=norm_type))

        self.compress_out = ConvBlock(self.num_groups * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)

    def forward(self, f_in, g1, g2):
        x = torch.cat((f_in, g1), dim=1)
        x = torch.cat((x, g2), dim=1)

        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            if idx == 2:
                x_mid = torch.cat((LD_L, g2), dim=1)
                LD_L = self.re_guide(x_mid)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)
        output = self.compress_out(output)

        return output


# ------build CFNet ------ #
class CFNet(nn.Module):
    def __init__(self, in_channels=args.in_channels, out_channels=args.out_channels, num_features=args.num_features,
                 num_steps=args.num_steps, upscale_factor=args.scale,
                 act_type=args.act_type,
                 norm_type=None,
                 num_cfbs=args.num_cfbs):
        super(CFNet, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        self.num_cfbs = num_cfbs

        # upscale_1
        self.upsample_over = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)

        # FEB_1
        self.conv_in_over = ConvBlock(in_channels, 4 * num_features, kernel_size=3, act_type=act_type,
                                      norm_type=norm_type)
        self.feat_in_over = ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type,
                                      norm_type=norm_type)

        # SRB_1
        self.srb_1 = SRB(norm_type)

        # REC_1
        self.out_over = DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                                    act_type='prelu', norm_type=norm_type)
        self.conv_out_over = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

        # upscale_2
        self.upsample_under = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)

        # FEB_2
        self.conv_in_under = ConvBlock(in_channels, 4 * num_features, kernel_size=3, act_type=None, norm_type=norm_type)
        self.feat_in_under = ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type,
                                       norm_type=norm_type)

        # SRB_2
        self.srb_2 = SRB(norm_type)

        # REC_2
        self.out_under = DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride,
                                     padding=padding,
                                     act_type=args.act_type, norm_type=norm_type)
        self.conv_out_under = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

        # CFBs and RECs
        self.CFBs_1 = []
        self.CFBs_2 = []
        self.out_1 = nn.ModuleList()
        self.conv_out_1 = nn.ModuleList()
        self.out_2 = nn.ModuleList()
        self.conv_out_2 = nn.ModuleList()

        for i in range(self.num_cfbs):
            cfb_over = 'cfb_over{}'.format(i)
            cfb_under = 'cfb_under{}'.format(i)
            cfb_1 = CFB(norm_type).cuda()
            cfb_2 = CFB(norm_type).cuda()
            setattr(self, cfb_over, cfb_1)
            self.CFBs_1.append(getattr(self, cfb_over))
            setattr(self, cfb_under, cfb_2)
            self.CFBs_2.append(getattr(self, cfb_under))

            self.out_1.append(
                DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                            act_type=args.act_type, norm_type=norm_type))
            self.conv_out_1.append(
                ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type))
            self.out_2.append(
                DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                            act_type=args.act_type, norm_type=norm_type))
            self.conv_out_2.append(
                ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type))

    # def forward(self, lr_over, lr_under):
    def forward(self, lr_over, lr_under):
        # upsampled version of input pairs
        up_over = self.upsample_over(lr_over)
        up_under = self.upsample_under(lr_under)

        # Feature extraction block
        f_in_over = self.conv_in_over(lr_over)
        f_in_over = self.feat_in_over(f_in_over)

        f_in_under = self.conv_in_under(lr_under)
        f_in_under = self.feat_in_under(f_in_under)

        # Super-resolution block
        g_over = self.srb_1(f_in_over)
        g_under = self.srb_2(f_in_under)

        # Coupled feedback block
        g_1 = [g_over]
        g_2 = [g_under]
        for i in range(self.num_cfbs):
            g_1.append(self.CFBs_1[i](f_in_over, g_1[i], g_2[i]))
            g_2.append(self.CFBs_2[i](f_in_under, g_2[i], g_1[i]))

        # Reconstruction
        res_1 = []
        res_2 = []
        res_over = self.out_over(g_over)
        res_over = self.conv_out_over(res_over)
        res_1.append(res_over)
        res_under = self.out_under(g_under)
        res_under = self.conv_out_under(res_under)
        res_2.append(res_under)
        for j in range(self.num_cfbs):
            res_o = self.out_1[j](g_1[j + 1])
            res_u = self.out_2[j](g_2[j + 1])
            res_1.append(self.conv_out_1[j](res_o))
            res_2.append(self.conv_out_2[j](res_u))

        # Output
        sr_over = []
        sr_under = []
        for k in range(self.num_cfbs + 1):
            image_over = torch.add(res_1[k], up_over)
            image_over = torch.clamp(image_over, -1.0, 1.0)
            image_over = (image_over + 1) * 127.5
            image_under = torch.add(res_2[k], up_under)
            image_under = torch.clamp(image_under, -1.0, 1.0)
            image_under = (image_under + 1) * 127.5
            sr_over.append(image_over)
            sr_under.append(image_under)

        return sr_over, sr_under
