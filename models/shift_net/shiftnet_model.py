import os
import time
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from util import util
from util.utils_howard import tensor2img, mkdir, enhance_img
from models import networks
from models.shift_net.base_model import BaseModel
from piqa import SSIM
import math
import cv2

class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.opt.input_nc == 3:
            self.loss_names = ['G_GAN', 'G_L1', 'D', 'style', 'content', 'tv', 'ssim']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs, if need d_score, add Ds
            self.model_names = ['G']

        # batchsize should be 1 for mask_global，初始化 mask
        self.mask_global = torch.zeros((self.opt.batchSize, 1, opt.loadSize, opt.loadSize), dtype=torch.bool)
        self.mask_global.zero_() # 填滿 0
        
        # 初始化預設 center
        self.mask_global[:, :, int(self.opt.loadSize/4): int(self.opt.loadSize/2) + int(self.opt.loadSize/4),\
                                int(self.opt.loadSize/4): int(self.opt.loadSize/2) + int(self.opt.loadSize/4)] = 1 # (white)

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
        
        use_sigmoid = False
        if opt.gan_type == 'vanilla': use_sigmoid = True  # only vanilla GAN using BCECriterion
        
        self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)

        # add style extractor
        if self.opt.input_nc == 3:
            self.vgg16_extractor = util.VGG16FeatureExtractor()
            if len(opt.gpu_ids) > 0:
                self.vgg16_extractor = self.vgg16_extractor.to(self.device)
        
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_mask = networks.Discounted_L1(opt).to(self.device) # make weights/buffers transfer to the correct device
            if self.opt.input_nc == 3:
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
            if self.opt.input_nc == 3:
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

        # create mask online (center)    
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.loadSize/4): int(self.opt.loadSize/2) + int(self.opt.loadSize/4),\
                            int(self.opt.loadSize/4): int(self.opt.loadSize/2) + int(self.opt.loadSize/4)] = 1
        self.rand_t, self.rand_l = int(self.opt.loadSize/4), int(self.opt.loadSize/4)
        # print(self.mask_global[0][torch.where(self.mask_global[0]==1)].size())

        self.set_latent_mask(self.mask_global)

        # masked_fill_(mask, value)，real_A 大小會跟 mask_global 一樣，將兩個疊起來，real_A 會將 mask_global 為 1 的位置取代為 0.(value)
        if self.opt.input_nc == 3:
            real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.) # R channel # gray ?
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

    def forward(self):
        self.set_gt_latent()
        start_time = time.time()
        self.fake_B = self.netG(self.real_A) # real_A 當 input 進去做 inpaint
        return time.time() - start_time

    '''
    for visualize rec difference or export patch
    '''
    def compute_diff(self, real_B, fake_B):
        if self.opt.measure_mode == 'MAE':
            l1 = torch.nn.L1Loss(reduction='none')
            return l1(real_B, fake_B)

        elif self.opt.measure_mode == 'MSE':
            return self.criterionL2(real_B, fake_B).detach().cpu().numpy()

        elif self.opt.measure_mode == 'SSIM':
            return self.criterionSSIM(torch.unsqueeze(real_B, 0), torch.unsqueeze(fake_B, 0)).detach().cpu().numpy()

        elif self.opt.measure_mode == 'Feat':
            self.netD.eval()
            with torch.no_grad():
                pred_fake = self.netD(fake_B)
                pred_real = self.netD(real_B)
            return self.criterionL2(pred_real, pred_fake).detach().cpu().numpy()

        elif self.opt.measure_mode == 'Style':
            score = 0.0
            vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B,0))
            vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B,0))
            for j in range(3):
                score += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[j]), util.gram_matrix(vgg_ft_realB[j]))
            return score  

        elif self.opt.measure_mode == 'Content':
            score = 0.0
            vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B,0))
            vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B,0))
            for j in range(3):
                score += self.criterionL2(vgg_ft_realB[j], vgg_ft_fakeB[j])
            return score  
        
        else:
            raise ValueError("Please choose one measure mode!")
        
    def visualize_diff(self, mode=None, fn=None):
        with torch.no_grad():
            model_pred_t = self.forward()
            print(f"model inpainting time cost: {model_pred_t}")

        fake_B = self.fake_B.detach() # Inpaint
        real_B = self.real_B # Original
        
        diff_B = self.compute_diff(real_B, fake_B) 
        
        gray_diff_B = rgb_to_grayscale(diff_B)
        
        patches, combine_t, denoise_t, export_t = self.combine_patches(fn, gray_diff_B, self.opt.results_dir, 'union')
        
        # self.export_inpaint_imgs(real_B, os.path.join(self.opt.results_dir, f'ori_diff_patches/{fn}'), 0) # 0 true, 1 fake
        # self.export_inpaint_imgs(fake_B, os.path.join(self.opt.results_dir, f'ori_diff_patches/{fn}'), 1) # 0 true, 1 fake
        # self.export_inpaint_imgs(patches, os.path.join(self.opt.results_dir, f'ori_diff_patches/{fn}'), 2) # 0 true, 1 fake

        return (model_pred_t, combine_t, denoise_t, export_t)
    
    def combine_patches(self, fn, patches, save_dir, overlap_strategy):
        if self.opt.isPadding:
            ORISIZE = 512
            EDGE_PIXEL = 6
            PADDING_PIXEL = 14
            IMGH = IMGW = (ORISIZE-2*EDGE_PIXEL+2*PADDING_PIXEL)
        else:
            IMGH = IMGW = 512
        
        start_time = time.time()
        if overlap_strategy == 'union':
            threshold = float(self.opt.binary_threshold)
            save_dir = os.path.join(save_dir, 'union')

            # thresholding
            patches[patches>=threshold] = 1
            patches[patches<threshold] = -1
            
            l = int(math.sqrt(patches.shape[0]))
            patches_reshape = patches.view(l,l,1,self.opt.loadSize,self.opt.loadSize)
            
            # create combined img template
            patches_combined = torch.zeros((1, IMGH, IMGW), device=self.device)

            # fill combined img
            ps = self.opt.loadSize # crop patch size
            sd = self.opt.crop_stride # crop stride
            idy = 0
            for y in range(0, IMGH, sd):
                # print(f"y {y}")
                if (y + ps) > IMGH:
                    break
                idx = 0
                for x in range(0, IMGW, sd):
                    # print(f"x {x}")
                    if (x + ps) > IMGW:
                        break
                    # 判斷是否 最上 y=0 & 最左=0 & 最右x=14
                    if idy == 0: # 只需考慮左右重疊
                        if idx == 0: # 最左邊直接先放
                            patches_combined[:, y:y+ps, x:x+ps] = patches_reshape[idy][idx]
                        else: 
                            # 左半聯集 
                            # 先相加，value only 1, -1 結果只會有 2, -2, 0
                            patches_combined[:, y:y+ps, x:x+(ps-sd)] = \
                                            patches_combined[:, y:y+ps, x:x+(ps-sd)] + patches_reshape[idy][idx][:, :, :(ps-sd)] 
                            # 0, 2 = 1 (or==True), -2 = -1 (or==False)
                            # print(patches_combined[:, y:y+ps, x:x+sd].shape)
                            # print(patches_combined[:, y:y+ps, x:x+sd])
                            # print(patches_combined[:, y:y+ps, x:x+sd].unique())
                            patches_combined[:, y:y+ps, x:x+(ps-sd)][patches_combined[:, y:y+ps, x:x+(ps-sd)]!=-2] = 1 # or=True 0,2
                            patches_combined[:, y:y+ps, x:x+(ps-sd)][patches_combined[:, y:y+ps, x:x+(ps-sd)]==-2] = -1  # or=False -2
                            # print(patches_combined[:, y:y+ps, x:x+sd].shape)
                            # print(patches_combined[:, y:y+ps, x:x+sd])
                            # print(patches_combined[:, y:y+ps, x:x+sd].unique())
                            # 右半，直接放
                            patches_combined[:, y:y+ps, x+(ps-sd):x+ps] = \
                                            patches_reshape[idy][idx][:, :, (ps-sd):ps]                           
                    else: # 還需考慮上下重疊
                        if idx == 0: 
                            # 上方聯集
                            patches_combined[:, y:y+(ps-sd), x:x+ps] = \
                                            patches_combined[:, y:y+(ps-sd), x:x+ps] + patches_reshape[idy][idx][:, :(ps-sd), :] 
                            patches_combined[:, y:y+(ps-sd), x:x+ps][patches_combined[:, y:y+(ps-sd), x:x+ps]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y:y+(ps-sd), x:x+ps][patches_combined[:, y:y+(ps-sd), x:x+ps]==-2] = -1  # or=False -2
                            # 下方，直接放
                            patches_combined[:, y+(ps-sd):y+ps, x:x+ps] = \
                                            patches_reshape[idy][idx][:, (ps-sd):ps, :]
                        else:
                            # 上方聯集
                            patches_combined[:, y:y+(ps-sd), x:x+ps] = \
                                            patches_combined[:, y:y+(ps-sd), x:x+ps] + patches_reshape[idy][idx][:, :(ps-sd), :] 
                            patches_combined[:, y:y+(ps-sd), x:x+ps][patches_combined[:, y:y+(ps-sd), x:x+ps]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y:y+(ps-sd), x:x+ps][patches_combined[:, y:y+(ps-sd), x:x+ps]==-2] = -1  # or=False -2
                            # 下左聯集
                            patches_combined[:, y+(ps-sd):y+ps, x:x+(ps-sd)] = \
                                            patches_combined[:, y+(ps-sd):y+ps, x:x+(ps-sd)] + patches_reshape[idy][idx][:, (ps-sd):ps, :(ps-sd)] 
                            patches_combined[:, y+(ps-sd):y+ps, x:x+(ps-sd)][patches_combined[:, y+(ps-sd):y+ps, x:x+(ps-sd)]!=-2] = 1 # or==True 0,2
                            patches_combined[:, y+(ps-sd):y+ps, x:x+(ps-sd)][patches_combined[:, y+(ps-sd):y+ps, x:x+(ps-sd)]==-2] = -1  # or=False -2
                            # 下右半直接放
                            patches_combined[:, y+(ps-sd):y+ps, x+(ps-sd):x+ps] = \
                                            patches_reshape[idy][idx][:, (ps-sd):ps, (ps-sd):ps]   
                    idx+=1
                idy+=1
            combine_t = time.time() - start_time
            print(f"combine time cost: {combine_t}")
            denoise_t=0
            start_time = time.time()
            # patches_combined = self.remove_small_areas(patches_combined, min_area)
            patches_combined = self.remove_small_areas_opencv(patches_combined)
            denoise_t = time.time() - start_time
            print(f"denoise time cost: {denoise_t}")
            
            if self.opt.isPadding:
                # crop flip part
                patches_combined = patches_combined[PADDING_PIXEL:-PADDING_PIXEL, PADDING_PIXEL:-PADDING_PIXEL]
                pad_width = ((EDGE_PIXEL, EDGE_PIXEL), (EDGE_PIXEL, EDGE_PIXEL))  # 上下左右各填充6个元素
                patches_combined = np.pad(patches_combined, pad_width, mode='constant', constant_values=0)
            
            start_time = time.time()
            # self.export_combined_diff_img(patches_combined, fn, os.path.join(save_dir, f'{threshold:.4f}_diff_pos_area_{min_area}/imgs'))
            self.export_combined_diff_img_opencv(patches_combined, fn, os.path.join(save_dir, f'{threshold:.4f}_diff_pos_area_{self.opt.min_area}/imgs'))
            export_t = time.time() - start_time
            print(f"export time cost: {export_t}")

        return patches, combine_t, denoise_t, export_t
    
    def remove_small_areas_opencv(self, image):
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
    
    def export_combined_diff_img_opencv(self, img, name, save_path):
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
                pil_img.save(os.path.join(save_path,f"{idx}.png"))
                pil_img_en = enhance_img(pil_img)
                pil_img_en.save(os.path.join(save_path,f"en_{idx}.png"))

    '''
    for compute rec anomaly score
    '''
    def compute_score(self, real_B, fake_B):
        if self.opt.measure_mode == 'MAE':
            return self.criterionL1(real_B, fake_B).detach().cpu().numpy()
        
        elif self.opt.measure_mode == 'Discounted_L1':
            return self.criterionL1_mask(real_B, fake_B).detach().cpu().numpy()

        elif self.opt.measure_mode == 'MSE':
            return self.criterionL2(real_B, fake_B).detach().cpu().numpy()

        elif self.opt.measure_mode == 'SSIM':
            return self.criterionSSIM(torch.unsqueeze(real_B, 0), torch.unsqueeze(fake_B, 0)).detach().cpu().numpy()

        elif self.opt.measure_mode == 'Feat':
            self.netD.eval()
            with torch.no_grad():
                pred_fake = self.netD(fake_B)
                pred_real = self.netD(real_B)
            return self.criterionL2(pred_real, pred_fake).detach().cpu().numpy()

        elif self.opt.measure_mode == 'Style':
            score = 0.0
            vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B,0))
            vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B,0))
            for j in range(3):
                score += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[j]), util.gram_matrix(vgg_ft_realB[j]))
            return score  

        elif self.opt.measure_mode == 'Content':
            score = 0.0
            vgg_ft_fakeB = self.vgg16_extractor(torch.unsqueeze(fake_B,0))
            vgg_ft_realB = self.vgg16_extractor(torch.unsqueeze(real_B,0))
            for j in range(3):
                score += self.criterionL2(vgg_ft_realB[j], vgg_ft_fakeB[j])
            return score  
        
        else:
            raise ValueError("Please choose one measure mode!")
    
    def test(self):
        with torch.no_grad():
            self.forward()
        fake_B = self.fake_B.detach() # Inpaint
        real_B = self.real_B # Original

        # ======Anomaly score======
        if self.opt.mask_part or self.opt.measure_mode == 'Discounted_L1':
            fake_B = fake_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                        self.rand_l:self.rand_l+self.opt.loadSize//2]
            real_B = real_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                        self.rand_l:self.rand_l+self.opt.loadSize//2]  
        crop_scores = []
        for i in range(0,real_B.shape[0]):
            crop_scores.append(self.compute_score(real_B[i], fake_B[i]))
        crop_scores = np.array(crop_scores)
        return crop_scores

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

    '''
    for train model
    '''
    def backward_D(self):
        fake_B = self.fake_B # Real
        
        real_B = self.real_B # GroundTruth

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            # Using the cropped fake_B as the input of D.
            fake_B = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                            self.rand_l:self.rand_l+self.opt.loadSize//2]

            real_B = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                            self.rand_l:self.rand_l+self.opt.loadSize//2]  
            
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
            fake_B = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                       self.rand_l:self.rand_l+self.opt.loadSize//2]
            real_B = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                       self.rand_l:self.rand_l+self.opt.loadSize//2]
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
            mask_patch_fake = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                                self.rand_l:self.rand_l+self.opt.loadSize//2]
            mask_patch_real = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.loadSize//2, \
                                                self.rand_l:self.rand_l+self.opt.loadSize//2]
                                        
            # Using Discounting L1 loss
            self.loss_G_L1_m += self.criterionL1_mask(mask_patch_fake, mask_patch_real)*self.opt.mask_weight_G # 400

        self.loss_G = self.loss_G_L1 + self.loss_G_L1_m + self.loss_G_GAN

        if self.opt.input_nc == 3:
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