from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import random
import inspect, re
import numpy as np
import os
import collections
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.transform import resize

# inMask is tensor should be bz*1*256*256 float
# Return: ByteTensor
def cal_feat_mask(inMask, nlayers):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    inMask = inMask.float()
    ntimes = 2**nlayers
    inMask = F.interpolate(inMask, (inMask.size(2)//ntimes, inMask.size(3)//ntimes), mode='nearest')
    inMask = inMask.detach().byte()

    return inMask

# It is only for patch_size=1 for now, although it also works correctly for patchsize > 1.
# For patch_size > 1, we adopt another implementation in `patch_soft_shift/innerPatchSoftShiftTripleModule.py` to get masked region.
# return: flag indicating where the mask is using 1s.
#         flag size: bz*(h*w)
def cal_flag_given_mask_thred(mask, patch_size, stride, mask_thred):
    assert mask.dim() == 4, "mask must be 4 dimensions"
    assert mask.size(1) == 1, "the size of the dim=1 must be 1"
    mask = mask.float()
    b = mask.size(0)
    mask = F.pad(mask, (patch_size//2, patch_size//2, patch_size//2, patch_size//2), 'constant', 0)
    m = mask.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    m = m.contiguous().view(b, 1, -1, patch_size, patch_size)
    m = torch.mean(torch.mean(m, dim=3, keepdim=True), dim=4, keepdim=True)
    mm = m.ge(mask_thred/(1.*patch_size**2)).long()
    flag = mm.view(b, -1)

    # Obsolete Method
    # It is Only for mask: H*W
    # dim = img.dim()
    # _, H, W = img.size(dim - 3), img.size(dim - 2), img.size(dim - 1)
    # nH = int(math.floor((H - patch_size) / stride + 1))
    # nW = int(math.floor((W - patch_size) / stride + 1))
    # N = nH * nW

    # flag = torch.zeros(N).long()
    # for i in range(N):
    #     h = int(math.floor(i / nW))
    #     w = int(math.floor(i % nW))
    #     mask_tmp = mask[h * stride:h * stride + patch_size,
    #                w * stride:w * stride + patch_size]

    #     if torch.sum(mask_tmp) < mask_thred:
    #         pass
    #     else:
    #         flag[i] = 1

    return flag

"""
   flow: N*h*w*2
        Indicating which pixel will shift to the location.
   mask: N*(h*w)
"""
def highlight_flow(flow, mask):
    """Convert flow into middlebury color code image.
    """
    assert flow.dim() == 4 and mask.dim() == 2
    assert flow.size(0) == mask.size(0)
    assert flow.size(3) == 2
    bz, h, w, _ = flow.shape
    out = torch.zeros(bz, 3, h, w).type_as(flow)
    for idx in range(bz):
        mask_index = (mask[idx] == 1).nonzero()
        img = torch.ones(3, h, w).type_as(flow) * 144.
        u = flow[idx, :, :, 0]
        v = flow[idx, :, :, 1]
        # It is quite slow here.
        for h_i in range(h):
            for w_j in range(w):
                p = h_i*w + w_j
                #If it is a masked pixel, we get which pixel that will replace it.
                # DO NOT USE `if p in mask_index:`, it is slow.
                if torch.sum(mask_index == p).item() != 0:
                    ui = u[h_i,w_j]
                    vi = v[h_i,w_j]
                    img[:, int(ui), int(vi)] = 255.
                    img[:, h_i, w_j] = 200. # Also indicating where the mask is.
        out[idx] = img
    return out

################# Style loss #########################
######################################################
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # print(self.enc_1)
        # print(self.enc_2)
        # print(self.enc_3)
        # raise
        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):

        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def gram_matrix(feat):
    (batch, ch, h, w) = feat.size()
    feat = feat.view(batch, ch, h*w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


