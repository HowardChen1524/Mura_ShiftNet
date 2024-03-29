import argparse
import os
from util import util, utils_howard
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Data
        parser.add_argument('--dataroot', type=str, help='Mura data path')
        parser.add_argument('--dataset_mode', type=str, default='aligned_sliding', help='[aligned_resized|aligned_sliding]')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset') 
        parser.add_argument('--loadSize', type=int, default=64, help='crop size')
        parser.add_argument('--mask_type', type=str, default='center', help='Mask 類型 [center|random]')
        parser.add_argument('--mask_sub_type', type=str, default='rect', help='Mask 形狀, [rect|fractal|island]')
        parser.add_argument('--add_mask2input', type=int, default=1, help='If True, It will add the mask as a fourth dimension over input space')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        
        # Model
        parser.add_argument('--model_version', type=str, help='model_version name')
        parser.add_argument('--model', type=str, default='shiftnet', help='chooses which model to use. [shiftnet]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD, [basic|densenet]')
        parser.add_argument('--which_model_netG', type=str, default='unet_shift_triple', help='selects model to use for netG [unet_256| unet_shift_triple| \
                                                                res_unet_shift_triple|patch_soft_unet_shift_triple| \
                                                                res_patch_soft_unet_shift_triple| face_unet_shift_triple]')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--bottleneck', type=int, default=512, help='neurals of fc')
        parser.add_argument('--mask_thred', type=int, default=1, help='number to decide whether a patch is masked')
        
        # shift layer
        parser.add_argument('--stride', type=int, default=1, help='should be dense, 1 is a good option.')
        parser.add_argument('--shift_sz', type=int, default=1, help='shift_sz>1 only for \'soft_shift_patch\'.')
        parser.add_argument('--show_flow', type=int, default=0, help='show the flow information. WARNING: set display_freq a large number as it is quite slow when showing flow')
        
        # loss & weight
        parser.add_argument('--norm', type=str, default='instance', help='[instance|batch|switchable] normalization')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')        
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')        
        parser.add_argument('--skip', type=int, default=0, help='Whether skip guidance loss, if skipped performance degrades with dozens of percents faster')
        parser.add_argument('--discounting', type=int, default=1, help='the loss type of mask part, whether using discounting l1 loss or normal l1')       
        parser.add_argument('--triple_weight', type=float, default=1, help='The weight on the gradient of skip connections from the gradient of shifted')
        parser.add_argument('--lambda_A', type=int, default=100, help='weight on L1 term in objective')
        parser.add_argument('--gp_lambda', type=float, default=10.0, help='gradient penalty coefficient')
        parser.add_argument('--constrain', type=str, default='MSE', help='guidance loss type')
        parser.add_argument('--strength', type=float, default=1, help='the weight of guidance loss')
        parser.add_argument('--fuse', type=int, default=0, help='Fuse may encourage large patches shifting when using \'patch_soft_shift\'')
        parser.add_argument('--gan_type', type=str, default='vanilla', help='wgan_gp, ' 'lsgan, ' 'vanilla, ' 're_s_gan (Relativistic Standard GAN), ')
        parser.add_argument('--gan_weight', type=float, default=0.2, help='the weight of gan loss')
        parser.add_argument('--style_weight', type=float, default=10.0, help='the weight of style loss')
        parser.add_argument('--content_weight', type=float, default=1.0, help='the weight of content loss')
        parser.add_argument('--tv_weight', type=float, default=0.0, help='the weight of tv loss, you can set a small value, such as 0.1/0.01')
        parser.add_argument('--mask_weight_G', type=float, default=400.0, help='the weight of mask part in ouput of G, you can try different mask_weight')
        parser.add_argument('--use_spectral_norm_D', type=int, default=1, help='whether to add spectral norm to D, it helps improve results')
        parser.add_argument('--use_spectral_norm_G', type=int, default=0, help='whether to add spectral norm in G. Seems very bad when adding SN to G')

        # save & load
        parser.add_argument('--checkpoints_dir', type=str, default='./log', help='models are saved here')
        parser.add_argument('--only_lastest', type=int, default=0, help='If True, it will save only the lastest weights')

        # other
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.model_version = opt.model_version + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, use \'-1 \' for cpu training/testing')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        
        # Howard add
        parser.add_argument('--resolution', type=str, default='origin', help='[origin, resized], default resize 512*512')
        parser.add_argument('--crop_stride', type=int, default=32, help='slding crop stride')
        parser.add_argument('--isPadding', type=int, default=0, help='whether to use padding')
        
        self.initialized = True # 表示已經做完初始化
        return parser

    def gather_options(self, options=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        if options == None:
            return parser.parse_args()
        else:
            return parser.parse_args(options)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

        # save to the disk -- problem, 只有 training 要取消註解
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.model_version)
        # utils_howard.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self, options=None):

        opt = self.gather_options(options=options)
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.model_version = opt.model_version + suffix

        self.print_options(opt)

        # set gpu ids
        if opt.gpu_ids != "-1":
            os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
        
        self.opt = opt
        return self.opt
