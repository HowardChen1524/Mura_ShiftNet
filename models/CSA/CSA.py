#-*-coding:utf-8-*-

import torch
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable

import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from .base_model import BaseModel
from . import networks
from .vgg16 import Vgg16

import os
import time
import math 
import cv2

from util.utils import mkdir, tensor2img, enhance_img
class CSA(BaseModel):
    def name(self):
        return 'CSAModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain


        self.vgg=Vgg16(requires_grad=False)
        self.vgg=self.vgg.cuda()
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.loadSize, opt.loadSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.loadSize, opt.loadSize)

        # batchsize should be 1 for mask_global
        self.mask_global = torch.BoolTensor(1, 1, opt.loadSize, opt.loadSize)


        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.loadSize/4) : int(self.opt.loadSize/2) + int(self.opt.loadSize/4), \
                                int(self.opt.loadSize/4) : int(self.opt.loadSize/2) + int(self.opt.loadSize/4)] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        self.netG,self.Cosis_list,self.Cosis_list2, self.CSA_model= networks.define_G(opt.input_nc_g, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        self.netP,_,_,_=networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netP, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion

            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netF,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)            
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netP, 'P', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_P)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netP)
            if self.isTrain:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
            print('-----------------------------------------------')
        else:
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

        assert (self.opt.resolution == 'resized') or (self.opt.resolution == 'origin')
        if self.opt.resolution == 'resized':
            self.RESOLUTION = (512,512)
        else:
            self.RESOLUTION = (1920,1080)

        if self.opt.isPadding:
            self.EDGE_PIXEL = 6
            self.PADDING_PIXEL = 14
            self.IMGH = self.RESOLUTION[1]-(2*self.EDGE_PIXEL)+(2*self.PADDING_PIXEL)
            self.IMGW = self.RESOLUTION[0]-(2*self.EDGE_PIXEL)+(2*self.PADDING_PIXEL)
        else:
            self.IMGH = self.RESOLUTION[1]
            self.IMGW = self.RESOLUTION[0]
            
        self.num_w_crop = math.ceil((self.IMGW-self.opt.loadSize)/self.opt.crop_stride) + 1
        self.num_h_crop = math.ceil((self.IMGH-self.opt.loadSize)/self.opt.crop_stride) + 1
        
    def set_input(self, input):
        
        input_A = input['A']
        input_B = input_A.clone()

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = 0

        if self.opt.mask_type == 'center':
            self.mask_global=self.mask_global
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)

        self.ex_mask = self.mask_global.expand(1, 3, self.mask_global.size(2), self.mask_global.size(3)) # 1*c*h*w

        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).bool()
        self.input_A.narrow(1,0,1).masked_fill_(self.mask_global, 2*123.0/255.0 - 1.0)
        self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)
        self.input_A.narrow(1,2,1).masked_fill_(self.mask_global, 2*117.0/255.0 - 1.0)

        self.set_latent_mask(self.mask_global, 3, self.opt.threshold)
        self.set_gt_latent()
        
    # It is quite convinient, as one forward-pass, all the innerCos will get the GT_latent!
    def set_latent_mask(self, mask_global, layer_to_last, threshold):
        self.CSA_model[0].set_mask(mask_global, layer_to_last, threshold)
        self.Cosis_list[0].set_mask(mask_global, self.opt)
        self.Cosis_list2[0].set_mask(mask_global, self.opt)
        
    def forward(self):
        start_time = time.time()
        self.real_A =self.input_A.to(self.device)
        self.fake_P= self.netP(self.real_A)
        self.un=self.fake_P.clone()
        self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
        self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
        self.Syn=self.Unknowregion+self.knownregion
        self.Middle=torch.cat((self.Syn,self.input_A),1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)
        return time.time() - start_time

    def set_gt_latent(self):
        gt_latent=self.vgg(Variable(self.input_B,requires_grad=False))
        self.Cosis_list[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent.relu4_3)

    def compute_score(self, real_B, fake_B):
        if self.opt.measure_mode == 'MAE':
            return self.criterionL1(real_B, fake_B).detach().cpu().numpy()
        elif self.opt.measure_mode == 'MSE':
            return self.criterionL2(real_B, fake_B).detach().cpu().numpy()
        else:
            raise ValueError("Please choose one measure mode!")
        
    def test(self):
        with torch.no_grad():
            t = self.forward()
            
            fake_B = self.fake_B.detach() # Inpaint
            real_B = self.real_B # Original

            # ======Anomaly score======
            if (self.opt.mask_part == 1):
                fake_B = fake_B[:, :, int(self.opt.loadSize/4) : int(self.opt.loadSize/2) + int(self.opt.loadSize/4), \
                                    int(self.opt.loadSize/4) : int(self.opt.loadSize/2) + int(self.opt.loadSize/4)]
                real_B = real_B[:, :, int(self.opt.loadSize/4) : int(self.opt.loadSize/2) + int(self.opt.loadSize/4), \
                                    int(self.opt.loadSize/4) : int(self.opt.loadSize/2) + int(self.opt.loadSize/4)]
            crop_scores = []
            for i in range(0, real_B.shape[0]):
                crop_scores.append(self.compute_score(real_B[i], fake_B[i]))
            crop_scores = np.array(crop_scores)
            return t, crop_scores

    def compute_diff(self, real_B, fake_B):
        if self.opt.measure_mode == 'MAE':
            l1 = torch.nn.L1Loss(reduction='none')
            return l1(real_B, fake_B)

        elif self.opt.measure_mode == 'MSE':
            l2 = torch.nn.MSELoss(reduction='none')
            return l2(real_B, fake_B)
        
        else:
            raise ValueError("Please choose one measure mode!")
        
    def visualize_diff(self, fn=None):
        model_pred_t, combine_t, denoise_t, export_t = 0,0,0,0

        with torch.no_grad():
            model_pred_t = self.forward()
            print(f"model inpainting time cost: {model_pred_t}")

        fake_B = self.fake_B.detach() # Inpaint
        real_B = self.real_B # Original
        
        diff_B = self.compute_diff(real_B, fake_B) 
        
        combine_t, denoise_t, export_t = self.create_visualize_image(fn, diff_B, self.opt.results_dir)

        self.export_inpaint_imgs(real_B, os.path.join(self.opt.results_dir, f'ori_diff_patches/{fn}'), 0) # 0 true, 1 fake
        self.export_inpaint_imgs(fake_B, os.path.join(self.opt.results_dir, f'ori_diff_patches/{fn}'), 1) # 0 true, 1 fake
        # self.export_inpaint_imgs(patches, os.path.join(self.opt.results_dir, f'ori_diff_patches/{fn}'), 2) # 0 true, 1 fake

        return (model_pred_t, combine_t, denoise_t, export_t)
    
    def create_visualize_image(self, fn, patches, save_dir):
        top_k = self.opt.top_k

        start_time = time.time()
        combine_t = time.time() - start_time
        patches_combined = self.combine_patches(patches)
        print(f"combine time cost: {combine_t}")

        patches_thesholding = self.top_percent_thesholding(patches_combined, top_k)

        start_time = time.time()
        patches_denoise = self.remove_small_areas(patches_thesholding)
        denoise_t = time.time() - start_time
        print(f"denoise time cost: {denoise_t}")

        if self.opt.isPadding:
            # crop flip part
            patches_denoise = patches_denoise[self.PADDING_PIXEL:-self.PADDING_PIXEL, self.PADDING_PIXEL:-self.PADDING_PIXEL]
            pad_width = ((self.EDGE_PIXEL, self.EDGE_PIXEL), (self.EDGE_PIXEL, self.EDGE_PIXEL))  # 上下左右各填充6个元素
            patches_res = np.pad(patches_denoise, pad_width, mode='constant', constant_values=0)
        
        start_time = time.time()
        self.export_combined_diff_img(patches_res, fn, os.path.join(save_dir, 'img'))
        export_t = time.time() - start_time
        print(f"export time cost: {export_t}")
            
        return combine_t, denoise_t, export_t
    
    def combine_patches(self, patches):
        image = torch.zeros((3, self.IMGH, self.IMGW), device=self.device)
        patches_count = torch.zeros((3, self.IMGH, self.IMGW), device=self.device)
        patches_reshape = patches.view(self.num_h_crop, self.num_w_crop, 3, self.opt.loadSize, self.opt.loadSize)
        ps = self.opt.loadSize # crop patch size
        sd = self.opt.crop_stride # crop stride
        idy = 0
        for idy in range(0, self.num_h_crop):
            crop_y = idy*sd
            if (idy*sd+ps) >= self.IMGH:
                crop_y = self.IMGH-ps
            for idx in range(0, self.num_w_crop):  
                crop_x = idx*sd
                if (idx*sd+ps) >= self.IMGW:
                    crop_x = self.IMGW-ps     
                image[:, crop_y:crop_y+ps, crop_x:crop_x+ps] += patches_reshape[idy][idx]
                patches_count[:, crop_y:crop_y+ps, crop_x:crop_x+ps] += 1.0

        image = image / patches_count
        
        image = rgb_to_grayscale(image)
        return image

    def top_percent_thesholding(self, image, top_k):
        # filter top five percent pixel value
        num_pixels = image.numel()
        num_top_pixels = math.ceil(num_pixels * top_k)
        filter, _ = image.view(-1).kthvalue(num_pixels - num_top_pixels)
        print(f"Theshold: {filter}")
        image[image>=filter] = 1
        image[image<filter] = -1
        return image

    def remove_small_areas(self, image):
        image = image.detach().cpu().numpy().transpose((1, 2, 0))
        image[image==-1] = 0
        image[image==1] = 255
        image = image.astype(np.uint8)
        
        # 使用 connectedComponents 函數
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        
        # 輸出連通區域的數量
        # print("連通區域的數量：", num_labels)

        # 輸出每個區域的統計資訊
        # for i in range(1, num_labels):
        #     left = stats[i, cv2.CC_STAT_LEFT]
        #     top = stats[i, cv2.CC_STAT_TOP]
        #     width = stats[i, cv2.CC_STAT_WIDTH]
        #     height = stats[i, cv2.CC_STAT_HEIGHT]
        #     area = stats[i, cv2.CC_STAT_AREA]
        #     print("區域", i, "的左上角座標：", (left, top), "寬度：", width, "高度：", height, "面積：", area)

        # 將每個區域的標籤轉換為彩色影像並顯示
        # label_hue = np.uint8(179 * labels / np.max(labels))
        # blank_ch = 255 * np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # labeled_img[label_hue == 0] = 0
        # cv2.imwrite('labeled_image.png', labeled_img)
        
        # 指定面積閾值
        min_area_threshold = self.opt.min_area
        
        # 遍歷所有區域
        for i in range(1, num_labels):
            # 如果區域面積小於閾值，就將對應的像素值設置為黑色
            if stats[i, cv2.CC_STAT_AREA] < min_area_threshold:
                labels[labels == i] = 0
        
        # 將標籤為 0 的像素設置為白色，其它像素設置為黑色
        result = labels.astype('uint8')
        # print(np.unique(labels))
        result[result == 0] = 0
        result[result != 0] = 255
        return result
    
    def export_combined_diff_img(self, img, name, save_path):
        mkdir(save_path)        
        cv2.imwrite(os.path.join(save_path, name), img)
        
    def export_inpaint_imgs(self, output, save_path, img_type):
        if img_type == 0:
            save_path =  os.path.join(save_path, 'real')
        elif img_type == 1:
            save_path =  os.path.join(save_path, 'fake')
        elif img_type == 2:
            save_path =  os.path.join(save_path, 'binary')
        mkdir(save_path)

        for idx, img in enumerate(output):
            pil_img = tensor2img(img) 
            if img_type == 2:
                pil_img = pil_img.convert('L')           
                pil_img.save(os.path.join(save_path,f"{idx}.png"))
            else:
                # pil_img.save(os.path.join(save_path,f"{idx}.png"))
                pil_img_en = enhance_img(pil_img)
                pil_img_en.save(os.path.join(save_path, f"en_{idx}.png"))



    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        self.gt_latent_fake = self.vgg(Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real = self.vgg(Variable(self.input_B, requires_grad=False))
        real_AB = self.real_B # GroundTruth




        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F,self.pred_real_F, True)

        self.loss_D =self.loss_D_fake * 0.5 + self.loss_F_fake  * 0.5

        # When two losses are ready, together backward.
        # It is op, so the backward will be called from a leaf.(quite different from LuaTorch)
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        fake_f = self.gt_latent_fake
        
        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(fake_f.relu3_3)
        
        pred_real=self.netD(self.real_B)
        pred_real_F=self.netF(self.gt_latent_real.relu3_3)

        self.loss_G_GAN = self.criterionGAN(pred_fake,pred_real, False)+self.criterionGAN(pred_fake_f, pred_real_F,False)

        # Second, G(A) = B
        self.loss_G_L1 =( self.criterionL1(self.fake_B, self.real_B) +self.criterionL1(self.fake_P, self.real_B) )* self.opt.lambda_A


        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight

        # Third add additional netG contraint loss!
        self.ng_loss_value = 0
        self.ng_loss_value2 = 0
        if self.opt.cosis:
            for gl in self.Cosis_list:
                #self.ng_loss_value += gl.backward()
                self.ng_loss_value += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value
            for gl in self.Cosis_list2:
                #self.ng_loss_value += gl.backward()
                self.ng_loss_value2 += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value2

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_G.zero_grad()
        self.optimizer_P.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_P.step()


    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D', self.loss_D_fake.data.item()),
                            ('F', self.loss_F_fake.data.item())
                            ])

    def get_current_visuals(self):

        real_A =self.real_A.data
        fake_B = self.fake_B.data
        real_B =self.real_B.data

        return real_A,real_B,fake_B


    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.gpu_ids)
        self.save_network(self.netP, 'P', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        self.save_network(self.netF, 'F', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
        self.load_network(self.netP, 'P', epoch)


