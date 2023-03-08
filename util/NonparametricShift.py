import random
import math
import numpy as np
import torch
import torch.nn as nn
from time import time

# These three functions only work when patch_size = 1x1
class Modified_NonparametricShift(object):

    def _extract_patches_from_flag(self, img, patch_size, stride, flag, value):
        input_windows = self._unfold(img, patch_size, stride)

        input_windows = self._filter(input_windows, flag, value)

        return self._norm(input_windows)

    # former: content, to be replaced.
    # latter: style, source pixels.
    def cosine_similarity(self, former, latter, patch_size, stride, flag, with_former=False):
        former_windows = self._unfold(former, patch_size, stride)
        former = self._filter(former_windows, flag, 1)

        latter_windows, i_2, i_3, i_1 = self._unfold(latter, patch_size, stride, with_indexes=True)
        latter = self._filter(latter_windows, flag, 0)

        num = torch.einsum('ik,jk->ij', [former, latter])
        norm_latter = torch.einsum("ij,ij->i", [latter, latter])
        norm_former = torch.einsum("ij,ij->i", [former, former])
        den = torch.sqrt(torch.einsum('i,j->ij', [norm_former, norm_latter]))
        if not with_former:
            return num / den, latter_windows, i_2, i_3, i_1
        else:
            return num / den, latter_windows, former_windows, i_2, i_3, i_1


    def _paste(self, input_windows, transition_matrix, i_2, i_3, i_1):
        ## TRANSPOSE FEATURES NEW FEATURES
        input_windows = torch.mm(transition_matrix, input_windows)

        ## RESIZE TO CORRET CONV FEATURES FORMAT
        input_windows = input_windows.view(i_2, i_3, i_1)
        input_windows = input_windows.permute(2, 0, 1).unsqueeze(0)
        return input_windows

    def _unfold(self,  img, patch_size, stride, with_indexes=False):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'

        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)

        i_1, i_2, i_3, i_4, i_5 = input_windows.size()

        if with_indexes:
            input_windows = input_windows.permute(1, 2, 0, 3, 4).contiguous().view(i_2 * i_3, i_1)
            return input_windows, i_2, i_3, i_1
        else:
            input_windows = input_windows.permute(1, 2, 0, 3, 4).contiguous().view(i_2 * i_3, i_1, i_4, i_5)
            return input_windows

    def _filter(self, input_windows, flag, value):
        ## EXTRACT MASK OR NOT DEPENDING ON VALUE
        input_window = input_windows[flag == value]
        return input_window.view(input_window.size(0), -1)


    def _norm(self, input_window):
        # This norm is incorrect.
        #return torch.norm(input_window, dim=1, keepdim=True)
        for i in range(input_window.size(0)):
            input_window[i] = input_window[i]*(1/(input_window[i].norm(2)+1e-8))

        return input_window

class Batch_NonShift(object):

    def _extract_patches_from_flag(self, img, patch_size, stride, flag, value):
        input_windows = self._unfold(img, patch_size, stride)

        input_windows = self._filter(input_windows, flag, value)

        return self._norm(input_windows)

    # former: content, to be replaced.
    # latter: style, source pixels.
    def cosine_similarity(self, former, latter, patch_size, stride, flag, with_former=False):
        former_windows = self._unfold(former, patch_size, stride)
        former = self._filter(former_windows, flag, 1)

        latter_windows, i_2, i_3, i_1 = self._unfold(latter, patch_size, stride, with_indexes=True)
        latter = self._filter(latter_windows, flag, 0)

        num = torch.einsum('bik,bjk->bij', [former, latter])
        norm_latter = torch.einsum("bij,bij->bi", [latter, latter])
        norm_former = torch.einsum("bij,bij->bi", [former, former])
        den = torch.sqrt(torch.einsum('bi,bj->bij', [norm_former, norm_latter]))
        if not with_former:
            return num / den, latter_windows, i_2, i_3, i_1
        else:
            return num / den, latter_windows, former_windows, i_2, i_3, i_1


    # delete i_4, as i_4 is 1
    def _paste(self, input_windows, transition_matrix, i_2, i_3, i_1):
        ## TRANSPOSE FEATURES NEW FEATURES
        bz = input_windows.size(0)
        input_windows = torch.bmm(transition_matrix, input_windows)

        ## RESIZE TO CORRET CONV FEATURES FORMAT
        input_windows = input_windows.view(bz, i_2, i_3, i_1)
        input_windows = input_windows.permute(0, 3, 1, 2)
        return input_windows

    def _unfold(self,  img, patch_size, stride, with_indexes=False):
        n_dim = 4
        assert img.dim() == n_dim, 'image must be of dimension 4.'

        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(2, kH, dH).unfold(3, kW, dW)

        i_0, i_1, i_2, i_3, i_4, i_5 = input_windows.size()

        if with_indexes:
            input_windows = input_windows.permute(0, 2, 3, 1, 4, 5).contiguous().view(i_0, i_2 * i_3, i_1)
            return input_windows, i_2, i_3, i_1
        else:
            input_windows = input_windows.permute(0, 2, 3, 1, 4, 5).contiguous().view(i_0, i_2 * i_3, i_1, i_4, i_5)
            return input_windows

    def _filter(self, input_windows, flag, value):
        ## EXTRACT MASK OR NOT DEPENDING ON VALUE
        assert flag.dim() == 2, "flag should be batch version"
        input_window = input_windows[flag == value]
        bz = flag.size(0)
        return input_window.view(bz, input_window.size(0)//bz, -1)

