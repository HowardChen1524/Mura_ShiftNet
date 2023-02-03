from collections import defaultdict
import torch
from torch.nn import functional as F
from util import util
from util.utils_howard import tensor2img, mkdir, enhance_img, plot_img_diff_hist
from models import networks
from models.shift_net.base_model import BaseModel
import time
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import cv2
from piqa import SSIM

class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'

    def create_random_mask(self):
        if self.opt.mask_type == 'random':
            if self.opt.mask_sub_type == 'fractal':
                assert 1==2, "It is broken somehow, use another mask_sub_type please"
                mask = util.create_walking_mask()  # create an initial random mask.

            elif self.opt.mask_sub_type == 'rect':
                mask, rand_t, rand_l = util.create_rand_mask(self.opt)
                self.rand_t = rand_t
                self.rand_l = rand_l
                return mask

            elif self.opt.mask_sub_type == 'island':
                mask = util.wrapper_gmask(self.opt)
        return mask

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain # train=True, test=False
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.opt.color_mode == 'RGB':
            self.loss_names = ['G_GAN', 'G_L1', 'D', 'style', 'content', 'tv', 'ssim']
            # self.loss_names = ['G_GAN', 'G_L1', 'D', 'style', 'content', 'tv']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.show_flow:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'flow_srcs']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs, if need d_score, add Ds
            self.model_names = ['G', 'D']


        # batchsize should be 1 for mask_global，初始化 mask
        self.mask_global = torch.zeros((self.opt.batchSize, 1, \
                                 opt.fineSize, opt.fineSize), dtype=torch.bool)
        self.mask_global.zero_() # 填滿 0
        
        # 初始化預設 center
        self.mask_global[:, :, int(self.opt.fineSize/4): int(self.opt.fineSize/2) + int(self.opt.fineSize/4),\
                                int(self.opt.fineSize/4): int(self.opt.fineSize/2) + int(self.opt.fineSize/4)] = 1

        if len(opt.gpu_ids) > 0:
            self.mask_global = self.mask_global.to(self.device)

        # load/define networks
        # self.ng_innerCos_list is the guidance loss list in netG inner layers.
        # self.ng_shift_list is the mask list constructing shift operation.
        # If add_mask2input=True, It will add the mask as a fourth dimension over input space
        if opt.add_mask2input:
            input_nc = opt.input_nc + 1
        else:
            input_nc = opt.input_nc

        self.netG, self.ng_innerCos_list, self.ng_shift_list = networks.define_G(input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_spectral_norm_G, opt.init_type, self.gpu_ids, opt.init_gain)
        
        # 06/19 add for generator and discriminator
        use_sigmoid = False
        if opt.gan_type == 'vanilla':
            use_sigmoid = True  # only vanilla GAN using BCECriterion
        # don't use cGAN
        self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)

        # add style extractor
        if self.opt.color_mode == 'RGB':
            self.vgg16_extractor = util.VGG16FeatureExtractor()
            if len(opt.gpu_ids) > 0:
                # self.vgg16_extractor = self.vgg16_extractor.to(self.gpu_ids[0])
                self.vgg16_extractor = self.vgg16_extractor.to(self.device)
                # self.vgg16_extractor = torch.nn.DataParallel(self.vgg16_extractor, self.gpu_ids) 多張 GPU 才需要
        
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_mask = networks.Discounted_L1(opt).to(self.device) # make weights/buffers transfer to the correct device
            if self.opt.color_mode == 'RGB':
                # VGG loss
                self.criterionL2_style_loss = torch.nn.MSELoss()
                self.criterionL2_content_loss = torch.nn.MSELoss()
                # TV loss
                self.tv_criterion = networks.TVLoss(self.opt.tv_weight)
            self.criterionSSIM = SSIM().to(self.device)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if self.opt.gan_type == 'wgan_gp':
                opt.beta1 = 0
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                    lr=opt.lr, betas=(opt.beta1, 0.9))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.9))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
        else:
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = SSIM().to(self.device)
            if self.opt.color_mode == 'RGB':
                # VGG loss
                self.criterionL2_style_loss = torch.nn.MSELoss()
                self.criterionL2_content_loss = torch.nn.MSELoss()
                # TV loss
                self.tv_criterion = networks.TVLoss(self.opt.tv_weight)
            self.criterionL1_mask = networks.Discounted_L1(opt).to(self.device)
            
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
            print('load model successful')
        self.print_networks(opt.verbose)

    def set_input(self, input):
        self.image_paths = input['A_paths']
        
        # print(input['A'].shape)
        real_A = input['A'].to(self.device)
        real_B = input['B'].to(self.device)
        
        # directly load mask offline
        # 使用 offline mask 才會有值，否則全為 0
        # torch.byte()将该tensor投射为byte类型
        self.mask_global = input['M'].to(self.device).byte() 
        # narrow(dim, index, size) : 表示取出tensor中第dim维上索引从index开始到index+size-1的所有元素存放在data中
        # torch.bool()将该tensor投射为bool类型
        self.mask_global = self.mask_global.narrow(1,0,1).bool() # 取 dim=1 (channel) 的 [0]
        # print(self.mask_global.shape)
        # raise

        # create mask online
        
        if self.opt.mask_type == 'center':
            self.mask_global.zero_()
            self.mask_global[:, :, int(self.opt.fineSize/4): int(self.opt.fineSize/2) + int(self.opt.fineSize/4),\
                                int(self.opt.fineSize/4): int(self.opt.fineSize/2) + int(self.opt.fineSize/4)] = 1
            self.rand_t, self.rand_l = int(self.opt.fineSize/4), int(self.opt.fineSize/4)
            # print(self.mask_global[0][torch.where(self.mask_global[0]==1)].size())
            
        elif self.opt.mask_type == 'random':
            # 函数tensor1.type_as(tensor2)将1的数据类型转换为2的数据类型
            self.mask_global = self.create_random_mask().type_as(self.mask_global).view(1, *self.mask_global.size()[-3:])
            # ***重要***
            # As generating random masks online are computation-heavy
            # So just generate one ranodm mask for a batch images. 
            self.mask_global = self.mask_global.expand(self.opt.batchSize, *self.mask_global.size()[-3:])
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)


        self.set_latent_mask(self.mask_global)

        # masked_fill_(mask, value)，real_A 大小會跟 mask_global 一樣，將兩個疊起來，real_A 會將 mask_global 為 1 的位置取代為 0.(value)
        if self.opt.color_mode == 'RGB':
            real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.) # R channel
            real_A.narrow(1,1,1).masked_fill_(self.mask_global, 0.) # G channel
            real_A.narrow(1,2,1).masked_fill_(self.mask_global, 0.) # B channel
        else:
            real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.) # gray channel

        if self.opt.add_mask2input:
            # # ***重要***
            # make it 4 dimensions.
            # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1. 但為啥會變 127 ???
            # ~（波浪符號），是反（Not）運算，會將 True 改成 False，False 改成 True, 挖空的部分原本是 1 -> 0
            real_A = torch.cat((real_A, (~self.mask_global).expand(real_A.size(0), 1, real_A.size(2), real_A.size(3)).type_as(real_A)), dim=1) 
            # expend 1,1,256,256
            # cat 在 第一維度 (channel) + mask = [(1,4,256,256) if RGB, (1,2,256,256) if gray]
            # type_as(real_A) 會從 bool -> float

        # 建立好 input real_A & real_B
        self.real_A = real_A
        self.real_B = real_B
    
    def set_latent_mask(self, mask_global):
        for ng_shift in self.ng_shift_list: # ITERATE OVER THE LIST OF ng_shift_list
            ng_shift.set_mask(mask_global)
        for ng_innerCos in self.ng_innerCos_list: # ITERATE OVER THE LIST OF ng_innerCos_list:
            ng_innerCos.set_mask(mask_global)

    def set_gt_latent(self):
        # 是否 skip guidance loss，預設 False
        if not self.opt.skip:
            if self.opt.add_mask2input:
                # make it 4 dimensions.
                # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
                real_B = torch.cat([self.real_B, (~self.mask_global).expand(self.real_B.size(0), 1, self.real_B.size(2), self.real_B.size(3)).type_as(self.real_B)], dim=1)
            else:
                real_B = self.real_B

            self.netG(real_B) # input ground truth

    def forward(self, mode=None, fn=None):
        self.set_gt_latent() # real_B，不知道幹嘛用
        self.fake_B = self.netG(self.real_A) # real_A 當 input 進去做 inpaint
        
        if ~(self.opt.isTrain) and (mode != None) and (fn != None):
            diff_B = torch.abs(self.real_B - self.fake_B)
            position_res_dir = os.path.join(f'/home/sallylab/Howard/detect_position/{self.opt.data_version}/{self.opt.crop_stride}')            

            self.export_inpaint_imgs(diff_B, mode, fn, os.path.join(position_res_dir, f'ori_diff_patches/{fn}'), 2) # 0 true, 1 fake
            self.combine_patches(fn, diff_B, position_res_dir, 'union')
            self.combine_patches(fn, diff_B, position_res_dir, 'mean')
            # self.export_inpaint_imgs(self.real_B, mode, fn, self.position_res_dir, 0) # 0 true, 1 fake
            # self.export_inpaint_imgs(self.fake_B, mode, fn, self.position_res_dir, 1) # 0 true, 1 fake

    def combine_patches(self, fn, patches, save_dir, overlap_strategy):
        if overlap_strategy == 'union':
            threshold = 0.0275
            save_dir = os.path.join(save_dir, 'union')
            mask_patches = patches[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            # plot patches threshold hist
            plot_img_diff_hist(patches.flatten().detach().cpu().numpy(), os.path.join(save_dir, f'all_patches_diff_hist/{fn}'), 0)
            plot_img_diff_hist(mask_patches.flatten().detach().cpu().numpy(), os.path.join(save_dir, f'all_patches_diff_hist/{fn}'), 1)

            # thresholding
            patches[patches>=threshold] = 1
            patches[patches<threshold] = -1

            # print(patches[224] == patches.reshape(15,15,3,64,64)[14][14])
            print(patches.shape)

            patches_reshape = patches.view(15,15,3,64,64)
            print(patches_reshape.shape)
            # create combined img template
            patches_combined = torch.zeros((3, 512, 512), device=self.device)
            print(patches_combined.shape)
            # fill combined img
            idy = 0
            for y in range(0, 512, self.opt.crop_stride): # stride default 32
                # print(f"y {y}")
                if (y + self.opt.fineSize) > 512:
                    break
                idx = 0
                for x in range(0, 512, self.opt.crop_stride):
                    # print(f"x {x}")
                    if (x + self.opt.fineSize) > 512:
                        break
                    # 判斷是否 最上 y=0 & 最左=0 & 最右x=14
                    if idy == 0: # 只需考慮左右重疊
                        if idx == 0: # 最左邊直接先放
                            patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.fineSize] = patches_reshape[idy][idx]
                        else: 
                            # 左半聯集 
                            # 先相加，value only 1, -1 結果只會有 2, -2, 0
                            patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride] = \
                                            patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride] + patches_reshape[idy][idx][:, :, :self.opt.crop_stride] 
                            # 0, 2 = 1 (or==True), -2 = -1 (or==False)
                            # print(patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride].shape)
                            # print(patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride])
                            # print(patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride].unique())
                            patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride][patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride][patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride]==-2] = -1  # or=False -2
                            # print(patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride].shape)
                            # print(patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride])
                            # print(patches_combined[:, y:y+self.opt.fineSize, x:x+self.opt.crop_stride].unique())
                            # 右半，直接放
                            patches_combined[:, y:y+self.opt.fineSize, x+self.opt.crop_stride:x+self.opt.fineSize] = \
                                            patches_reshape[idy][idx][:, :, self.opt.crop_stride:self.opt.fineSize]                           
                    else: # 還需考慮上下重疊
                        if idx == 0: 
                            # 上方聯集
                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize] = \
                                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize] + patches_reshape[idy][idx][:, :self.opt.crop_stride, :] 
                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize][patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize][patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize]==-2] = -1  # or=False -2
                            # 下方，直接放
                            patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.fineSize] = \
                                            patches_reshape[idy][idx][:, self.opt.crop_stride:self.opt.fineSize, :]
                        else:
                            # 上方聯集
                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize] = \
                                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize] + patches_reshape[idy][idx][:, :self.opt.crop_stride, :] 
                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize][patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize][patches_combined[:, y:y+self.opt.crop_stride, x:x+self.opt.fineSize]==-2] = -1  # or=False -2
                            # 下左聯集
                            patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.crop_stride] = \
                                            patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.crop_stride] + patches_reshape[idy][idx][:, self.opt.crop_stride:self.opt.fineSize, :self.opt.crop_stride] 
                            patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.crop_stride][patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.crop_stride]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.crop_stride][patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x:x+self.opt.crop_stride]==-2] = -1  # or=False -2
                            # 下右半直接放
                            patches_combined[:, y+self.opt.crop_stride:y+self.opt.fineSize, x+self.opt.crop_stride:x+self.opt.fineSize] = \
                                            patches_reshape[idy][idx][:, self.opt.crop_stride:self.opt.fineSize, self.opt.crop_stride:self.opt.fineSize]   
                    idx+=1
                idy+=1
            
            self.export_combined_diff_img(patches_combined, fn, os.path.join(save_dir, f'{threshold:.4f}_diff_pos/'))
        elif overlap_strategy == 'mean':
            pass

    def export_combined_diff_img(self, img, name, save_path):
        mkdir(save_path)       
        pil_img = tensor2img(img) 
        pil_img = pil_img.convert('L')           
        pil_img.save(os.path.join(save_path, name))


    def export_inpaint_imgs(self, output, mode, name, save_path, img_type):
        if img_type == 0:
            save_path =  os.path.join(save_path, 'real')
        elif img_type == 1:
            save_path =  os.path.join(save_path, 'fake')
        elif img_type == 2:
            save_path =  os.path.join(save_path, 'diff')

        mkdir(save_path)
        for idx, img in enumerate(output):
            pil_img = tensor2img(img) 
            # pil_img = pil_img.convert('L')           
            pil_img.save(os.path.join(save_path,f"{idx}.png"))
            pil_img_en = enhance_img(pil_img)
            pil_img_en.save(os.path.join(save_path,f"en_{idx}.png"))

    # 06/18 add for testing
    def test(self, mode=None, fn=None):
        # ======Inpainting method======
        if self.opt.inpainting_mode == 'ShiftNet':
            # torch.no_grad() disables the gradient calculation，等於 torch.set_grad_enabled(False)                       
            with torch.no_grad():
                self.forward(mode, fn)
            fake_B = self.fake_B.detach() # Inpaint
            real_B = self.real_B # Original

        elif self.opt.inpainting_mode == 'OpenCV' and self.opt.color_mode != 'RGB':
            real_B = self.real_B # Original
            # print(real_B)
            # print(real_B.shape)
            # =====Create fake_B=====
            
            # 去掉第一個 batch dim (1,1,h,w)
            mask_global = np.copy(self.mask_global.detach().cpu().numpy()[0]) 
            fake_B = np.copy(self.real_B.detach().cpu().numpy()[0])

            # 建立 mask
            mask = np.zeros(mask_global.shape)
            mask[np.where(mask_global==True)] = 255 # white
            mask[np.where(mask_global!=True)] = 0 # black      
            # print(mask[np.where(mask==255)].shape)

            mask = np.transpose(mask, (1, 2, 0)) # (1,h,w) -> (h,w,1)
            mask = np.squeeze(mask, axis=2) # (h,w,1) -> (h,w)
            # mask = mask.astype('uint8')
            # mask_img = Image.fromarray(mask)
            # mask_img.save('./opencv_mask.png')
            
            # 建立 fake_B
            fake_B = (fake_B + 1) / 2.0 * 255.0
            fake_B[np.where(mask_global==True)] = 0 # black
            # print(fake_B[np.where(fake_B==0)].shape)

            fake_B = np.transpose(fake_B, (1, 2, 0))
            fake_B = np.squeeze(fake_B, axis=2)
            fake_B = fake_B.astype('float32')
            fake_B = cv2.inpaint(fake_B, mask, 3, cv2.INPAINT_NS)          
            # fake_B_img = Image.fromarray(fake_B.astype('uint8'))
            # fake_B_img.save('./opencv_img.png')
            fake_B = (fake_B / 255.0 * 2) - 1
            
            # change back tensor
            fake_B = torch.from_numpy(fake_B)
            # transform = transforms.Compose([transforms.ToTensor(),
            #                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
            # fake_B = transform(fake_B_img).to(self.device)
            fake_B = fake_B.view(1,1,64,64).to(self.device)
            # print(fake_B)
            # print(fake_B.shape)
        elif self.opt.inpainting_mode == 'Mean' and self.opt.color_mode != 'RGB':
            real_B = self.real_B # Original
            # print(real_B)
            # =====Create fake_B=====
            # 去掉第一個 batch dim (1,1,256,256)
            mask_global = np.copy(self.mask_global.detach().cpu().numpy()[0])
            fake_B = np.copy(self.real_B.detach().cpu().numpy()[0])

            # 建立 fake_B
            mean = fake_B[np.where(mask_global!=True)].mean()
            # fake_B[np.where(mask_global==True)] = -1 # white
            fake_B[np.where(mask_global==True)] = mean
            
            fake_B = np.transpose(fake_B, (1, 2, 0))
            fake_B = np.squeeze(fake_B, axis=2)
            fake_B_img = Image.fromarray(fake_B.astype('uint8'))
            fake_B_img.save('./mean_img.png')

            fake_B = torch.from_numpy(fake_B)
            fake_B = fake_B.view(1,1,64,64).to(self.device)
            
            # print(fake_B)
        else:
            raise ValueError("Please choose one inpainting mode!")

        # ======Anomaly score======
        if self.opt.measure_mode == 'MSE':
            return self.criterionL2(real_B, fake_B).detach().cpu().numpy()

        elif self.opt.measure_mode == 'Mask_MSE':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]  
            return self.criterionL2(real_B, fake_B).detach().cpu().numpy()
        
        elif self.opt.measure_mode == 'MAE_sliding':
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL1(real_B[i], fake_B[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores    

        elif self.opt.measure_mode == 'Mask_MAE_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]  
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL1(real_B[i], fake_B[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores
        
        elif self.opt.measure_mode == 'Discounted_L1_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2]
            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2]
                                        
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL1_mask(real_B[i], fake_B[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores
            
        elif self.opt.measure_mode == 'MSE_sliding':
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL2(real_B[i], fake_B[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores    

        elif self.opt.measure_mode == 'Mask_MSE_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]  
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL2(real_B[i], fake_B[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores

        elif self.opt.measure_mode == 'SSIM_sliding':
            crop_scores = []
            for i in range(0,225):
                crop_scores.append((self.criterionSSIM(torch.unsqueeze(real_B[i], 0), torch.unsqueeze(fake_B[i], 0))).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores   

        elif self.opt.measure_mode == 'Mask_SSIM_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]  
            crop_scores = []
            for i in range(0,225):
                crop_scores.append((self.criterionSSIM(torch.unsqueeze(real_B[i], 0), torch.unsqueeze(fake_B[i], 0))).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores
 
        elif self.opt.measure_mode == 'MSE_SSIM_sliding':
            MSE_crop_scores = []
            SSIM_crop_scores = []
            for i in range(0,225):
                MSE_crop_scores.append(self.criterionL2(real_B[i], fake_B[i]).detach().cpu().numpy())
                SSIM_crop_scores.append((1-self.criterionSSIM(torch.unsqueeze(real_B[i], 0), torch.unsqueeze(fake_B[i], 0))).detach().cpu().numpy())
            crop_scores = defaultdict()
            crop_scores['MSE'] = np.array(MSE_crop_scores)
            crop_scores['SSIM'] = np.array(SSIM_crop_scores)
            return crop_scores   

        elif self.opt.measure_mode == 'Mask_MSE_SSIM_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]  
            MSE_crop_scores = []
            SSIM_crop_scores = []
            for i in range(0,225):
                MSE_crop_scores.append(self.criterionL2(real_B[i], fake_B[i]).detach().cpu().numpy())
                SSIM_crop_scores.append((1-self.criterionSSIM(torch.unsqueeze(real_B[i], 0), torch.unsqueeze(fake_B[i], 0))).detach().cpu().numpy())
            crop_scores = defaultdict()
            crop_scores['MSE'] = np.array(MSE_crop_scores)
            crop_scores['SSIM'] = np.array(SSIM_crop_scores)
            return crop_scores   

        elif self.opt.measure_mode == 'ADV_sliding':
            self.netD.eval()
            with torch.no_grad():
                pred_fake = self.netD(fake_B)

            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionGAN(pred_fake[i], True).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Mask_ADV_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            self.netD.eval()
            with torch.no_grad():
                pred_fake = self.netD(fake_B)

            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionGAN(pred_fake[i], True).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Dis_sliding':
            self.netD.eval()
            with torch.no_grad():
                pred_fake = self.netD(fake_B)
                pred_real = self.netD(real_B)
            # print(pred_fake.shape)
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL2(pred_real[i], pred_fake[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Mask_Dis_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]
            self.netD.eval()
            with torch.no_grad():
                pred_fake = self.netD(fake_B)
                pred_real = self.netD(real_B)
            print(pred_fake.shape)
            crop_scores = []
            for i in range(0,225):
                crop_scores.append(self.criterionL2(pred_real[i], pred_fake[i]).detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores

        elif self.opt.measure_mode == 'Style_VGG16_sliding':
            crop_scores = []
            for i in range(0,225):
                score = 0.0
                vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B[i],0))
                vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B[i],0))
                for j in range(3):
                    score += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[j]), util.gram_matrix(vgg_ft_realB[j]))
                crop_scores.append(score.detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Mask_Style_VGG16_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]
            crop_scores = []
            for i in range(0,225):
                score = 0.0
                vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B[i],0))
                vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B[i],0))
                for j in range(3):
                    score += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[j]), util.gram_matrix(vgg_ft_realB[j]))
                crop_scores.append(score.detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Content_VGG16_sliding':
            crop_scores = []
            for i in range(0,225):
            # for i in range(0,64):
                score = 0.0
                vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B[i],0))
                vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B[i],0))
                for j in range(3):
                    score += self.criterionL2(vgg_ft_realB[j], vgg_ft_fakeB[j])
                crop_scores.append(score.detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Mask_Content_VGG16_sliding':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]
            crop_scores = []
            for i in range(0,225):
                score = 0.0
                vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B[i],0))
                vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B[i],0))
                for j in range(3):
                    score += self.criterionL2(vgg_ft_realB[j], vgg_ft_fakeB[j])
                crop_scores.append(score.detach().cpu().numpy())
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'G_loss_combined':
            crop_scores = []

            mask_fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2]
            mask_real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2]
            print(self.opt.gan_weight)
            print(self.opt.lambda_A)
            print(self.opt.mask_weight_G)
            print(self.opt.style_weight)
            print(self.opt.content_weight)
            for i in range(0,225):
                score = 0
                
                # # adv
                # self.netD.eval()
                # with torch.no_grad():
                #     pred_fake = self.netD(fake_B[i])
                # adv_loss = (self.criterionGAN(pred_fake, True)*self.opt.gan_weight).detach().cpu().numpy()

                # L1
                l1_loss = (self.criterionL1(real_B[i], fake_B[i])*self.opt.lambda_A).detach().cpu().numpy()
                
                # D L1
                D_l1_loss = (self.criterionL1_mask(mask_real_B[i], mask_fake_B[i])*self.opt.mask_weight_G).detach().cpu().numpy()

                # SSIM
                SSIM_loss = (1-self.criterionSSIM(torch.unsqueeze(real_B[i], 0), torch.unsqueeze(fake_B[i], 0))).detach().cpu().numpy()

                # Content loss & style loss
                content_loss, style_loss = 0, 0
                vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B[i],0))
                vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B[i],0))
                for j in range(3):
                    style_loss += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[j]), util.gram_matrix(vgg_ft_realB[j]))
                    content_loss += self.criterionL2(vgg_ft_realB[j], vgg_ft_fakeB[j])
                style_loss = (style_loss*self.opt.style_weight).detach().cpu().numpy() 
                content_loss = (content_loss*self.opt.content_weight).detach().cpu().numpy()
                
                score = l1_loss + D_l1_loss + SSIM_loss + style_loss + content_loss

                crop_scores.append(score)
            crop_scores = np.array(crop_scores)
            return crop_scores

        elif self.opt.measure_mode == 'R_square':
            crop_scores = []
            for i in range(0,225):
                crop_scores.append((1-(self.criterionL2(real_B[i], fake_B[i])/torch.var(real_B[i], unbiased=False))).detach().cpu().numpy())
            
            crop_scores = np.array(crop_scores)
            return crop_scores  

        elif self.opt.measure_mode == 'Mask_R_square':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            crop_scores = []
            for i in range(0,225):
                print(self.criterionL2(real_B[i], fake_B[i]))
                print(torch.var(real_B[i], unbiased=False))
                crop_scores.append((1-(self.criterionL2(real_B[i], fake_B[i])/torch.var(real_B[i], unbiased=False))).detach().cpu().numpy())
                print(crop_scores)
                raise
            crop_scores = np.array(crop_scores)
            return crop_scores 

        else:
            raise ValueError("Please choose one measure mode!")

    # Just assume one shift layer.
    def set_flow_src(self):
        self.flow_srcs = self.ng_shift_list[0].get_flow()
        self.flow_srcs = F.interpolate(self.flow_srcs, scale_factor=8, mode='nearest')
        # Just to avoid forgetting setting show_map_false
        self.set_show_map_false()

    # Just assume one shift layer.
    def set_show_map_true(self):
        self.ng_shift_list[0].set_flow_true()

    def set_show_map_false(self):
        self.ng_shift_list[0].set_flow_false()

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_B = self.fake_B # Real
        
        real_B = self.real_B # GroundTruth

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            # Using the cropped fake_B as the input of D.
            fake_B = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]

            real_B = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2]  
            
        self.pred_fake = self.netD(fake_B.detach())
        self.pred_real = self.netD(real_B)

        if self.opt.gan_type == 'wgan_gp':
            gradient_penalty, _ = util.cal_gradient_penalty(self.netD, real_B, fake_B.detach(), self.device, constant=1, lambda_gp=self.opt.gp_lambda)
            self.loss_D_fake = torch.mean(self.pred_fake)
            self.loss_D_real = -torch.mean(self.pred_real)

            self.loss_D = self.loss_D_fake + self.loss_D_real + gradient_penalty
        else:
            # default
            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_D_fake = self.criterionGAN(self.pred_fake, False) # default BCE loss
                self.loss_D_real = self.criterionGAN(self.pred_real, True)

                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            elif self.opt.gan_type == 're_s_gan':
                self.loss_D = self.criterionGAN(self.pred_real - self.pred_fake, True)

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = self.fake_B
        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            fake_B = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                       self.rand_l:self.rand_l+self.opt.fineSize//2]
            real_B = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                       self.rand_l:self.rand_l+self.opt.fineSize//2]
        else:
            real_B = self.real_B

        pred_fake = self.netD(fake_B)

        # adverial loss
        if self.opt.gan_type == 'wgan_gp':
            self.loss_G_GAN = -torch.mean(pred_fake)
        else:
            # default
            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.gan_weight 

            elif self.opt.gan_type == 're_s_gan':
                pred_real = self.netD(real_B)
                self.loss_G_GAN = self.criterionGAN (pred_fake - pred_real, True) * self.opt.gan_weight

            elif self.opt.gan_type == 're_avg_gan':
                self.pred_real = self.netD(real_B)
                self.loss_G_GAN =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), False) \
                               + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), True)) / 2.
                self.loss_G_GAN *=  self.opt.gan_weight


        # If we change the mask as 'center with random position', then we can replacing loss_G_L1_m with 'Discounted L1'.
        # l1 loss, Discounting L1 loss
        self.loss_G_L1, self.loss_G_L1_m = 0, 0
        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A # 100
        # calcuate mask construction loss
        # When mask_type is 'center' or 'random_with_rect', we can add additonal mask region construction loss (traditional L1).
        # Only when 'discounting_loss' is 1, then the mask region construction loss changes to 'discounting L1' instead of normal L1.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            mask_patch_fake = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2]
            mask_patch_real = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2]
                                        
            # Using Discounting L1 loss
            self.loss_G_L1_m += self.criterionL1_mask(mask_patch_fake, mask_patch_real)*self.opt.mask_weight_G # 400

        self.loss_G = self.loss_G_L1 + self.loss_G_L1_m + self.loss_G_GAN

        if self.opt.color_mode == 'RGB':
            # Then, add TV loss
            self.loss_tv = self.tv_criterion(self.fake_B*self.mask_global.float())

            # Finally, add style loss
            vgg_ft_fakeB = self.vgg16_extractor(fake_B)
            vgg_ft_realB = self.vgg16_extractor(real_B)
            self.loss_style = 0
            self.loss_content = 0

            for i in range(3):
                self.loss_style += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[i]), util.gram_matrix(vgg_ft_realB[i]))
                self.loss_content += self.criterionL2_content_loss(vgg_ft_fakeB[i], vgg_ft_realB[i])

            self.loss_style *= self.opt.style_weight
            self.loss_content *= self.opt.content_weight

            self.loss_G += (self.loss_style + self.loss_content + self.loss_tv)

        # 08/23 new add SSIM loss
        self.loss_ssim = 0
        self.loss_ssim = 1-self.criterionSSIM(self.fake_B, self.real_B)
        self.loss_G += self.loss_ssim
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward() # forward propagation

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad() # 清空 gradient
        self.backward_D() # back propagation
        self.optimizer_D.step() # gradient update

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()