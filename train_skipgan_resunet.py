import os
import random
import itertools
from tqdm import tqdm
from collections import defaultdict

from packaging import version

import numpy as np
import pandas as pd 
import cv2
from PIL import Image, ImageEnhance

# import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from options.train_options import TrainOptions
from glob import glob

import torch
print(f'\nAvailable cuda = {torch.cuda.is_available()}')
print(f'\nGPUs availables = {torch.cuda.device_count()}')
print(f'\nCurrent device = {torch.cuda.current_device()}')
print(f'\nCurrent Device location = {torch.cuda.device(0)}')
print(f'\nName of the device = {torch.cuda.get_device_name(0)}')
import tensorflow as tf
print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')

# ===== initalize =====
opt = TrainOptions().parse()
if opt.resolution == 'resized':
    ORI_SIZE = (512, 512)  
else:
    ORI_SIZE = (1080, 1920)  
IMG_H, IMG_W, IMG_C  = opt.loadSize, opt.loadSize, 3
winSize = (opt.loadSize, opt.loadSize)
stSize = opt.crop_stride
EDGE_PIXEL = 6
PADDING_PIXEL = 14
LIMIT_TEST_IMAGES = "MAX"
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# Weight initializers for the Generator network
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
AUTOTUNE = tf.data.AUTOTUNE

# ===== loss functions =====
class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self,
         reduction=tf.keras.losses.Reduction.AUTO,
         name='SSIMLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, ori, recon):
        recon = tf.convert_to_tensor(recon)
        ori = tf.cast(ori, recon.dtype)

        loss_ssim = tf.reduce_mean(1 - tf.image.ssim(ori, recon, max_val=IMG_W, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))
        return loss_ssim

class MultiFeatureLoss(tf.keras.losses.Loss):
    def __init__(self,
             reduction=tf.keras.losses.Reduction.AUTO,
             name='FeatureLoss'):
        super().__init__(reduction=reduction, name=name)
        self.mse_func = tf.keras.losses.MeanSquaredError() 

    
    def call(self, real, fake, weight=1):
        result = 0.0
        for r, f in zip(real, fake):
            result = result + (weight * self.mse_func(r, f))
        
        return result

# for adversarial loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# L1 Loss
mae = tf.keras.losses.MeanAbsoluteError()
# L2 Loss
mse = tf.keras.losses.MeanSquaredError() 
multimse = MultiFeatureLoss()
# SSIM loss
ssim = SSIMLoss()

# ===== preprocessing =====
def sliding_crop(img, stepSize=stSize, windowSize=winSize):
    current_image = []
    y_end_crop = False
    (h, w, _) = img.shape
    
    for y in range(0,h, stepSize):
        if y_end_crop: break
        y_end_crop = False
        crop_y = y
        if (y + windowSize[0]) >= h:
            crop_y =  h - windowSize[0]
            y_end_crop = True
        x_end_crop = False
        for x in range(0, w, stepSize):
            if x_end_crop: break
            crop_x = x
            if (x + windowSize[1]) >= w:
                crop_x = w - windowSize[1]
                x_end_crop = True
            # print(crop_y, crop_x, windowSize)
            image = tf.image.crop_to_bounding_box(img, crop_y, crop_x, windowSize[0], windowSize[1])
            current_image.append(image)
    current_image = shuffle(current_image, random_state=random.randint(123, 10000))
    current_image = current_image[:opt.crop_image_num]
    return current_image

def prep_stage(x):
    x = tf.image.resize(x, ORI_SIZE) # Howard
   
    return x

def post_stage(x):
    # normalize to the range -1,1
    x = tf.cast(x, tf.float32)
    x = (x - 127.5) / 127.5
    # normalize to the range 0-1
    # img /= 255.0
    return x

# ===== dataset =====
def load_image_train(filename, batch_size):    
    pixels = tf_dataset(filename, batch_size)
    
    return pixels

def tf_dataset(images_path, batch_size):
    
    images_path = shuffle(images_path, random_state=random.randint(123, 10000))
    
    if opt.fix_step != "MAX":
        images_path = images_path[:opt.fix_step]
        
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=IMG_C)
    img = prep_stage(img)
    
    img_list = sliding_crop(img) # Howard
    img = post_stage(img_list)
    return img

# ===== model =====
def build_skipgan_generator(input_shape):
    
    conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_shape)
    # conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    # conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool3)
    # conv4 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = tf.keras.layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = tf.keras.layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(conv5)
    
    
    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
    # conv7 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
    # conv8 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
    # conv9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(conv9)
    
    model = tf.keras.models.Model(inputs, conv10)

    return model

def build_skipgan_discriminator(inputs):
    num_layers = 4
    f = [2**i for i in range(num_layers)]
    x = inputs
    
    for i in range(0, num_layers):
        if i == 0:
            x = tf.keras.layers.Conv2D(f[i] * 512 ,kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        else:
            x = tf.keras.layers.Conv2D(f[i] * 512 ,kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            # x = tf.keras.layers.Dropout(0.3)(x)      
    
    x = tf.keras.layers.Flatten()(x)
    features = x
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.models.Model(inputs, outputs = [features, output])
    return model

class SkipGanomaly(tf.keras.models.Model):
    def __init__(self, generator, discriminator):
        super(SkipGanomaly, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
       
        # Regularization Rate for each loss function
        self.ADV_REG_RATE_LF = 1
        self.REC_REG_RATE_LF = 50
        self.FEAT_REG_RATE_LF = 1
        self.field_names = ['epoch', 'gen_loss', 'disc_loss']
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, beta_1=0.5, beta_2=0.999)
    
    def compile(self, g_optimizer, d_optimizer):
        super(SkipGanomaly, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print("Images: ", images)
            reconstructed_images = self.generator(images, training=True)
            feature_real, label_real = self.discriminator(images, training=True)
            # print(generated_images.shape)
            feature_fake, label_fake = self.discriminator(reconstructed_images, training=True)
            # Loss 1: ADVERSARIAL loss
            
            real_loss = cross_entropy(label_real, tf.ones_like(label_real))
            fake_loss = cross_entropy(label_fake, tf.zeros_like(label_fake))
            disc_adv_loss = real_loss + fake_loss
            
            gen_adv_loss = cross_entropy(label_fake, tf.ones_like(label_real))
            
            # Loss 2: RECONSTRUCTION loss (L1)
            loss_rec = mae(images, reconstructed_images)
        
            # Loss 3: FEATURE Loss
            loss_feat = mse(feature_real, feature_fake)
            
            gen_loss = tf.reduce_mean( 
                (gen_adv_loss * self.ADV_REG_RATE_LF) 
                + (loss_rec * self.REC_REG_RATE_LF) 
                + (loss_feat * self.FEAT_REG_RATE_LF) 
            )
            
            disc_loss = tf.reduce_mean( (disc_adv_loss * self.ADV_REG_RATE_LF) + (loss_feat * self.FEAT_REG_RATE_LF) )


        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        

        
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        


        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "gen_adv_loss": gen_adv_loss,
            "disc_adv_loss": disc_adv_loss,
            "loss_rec": loss_rec,
            "loss_feat": loss_feat
        }

    def saved_model(self, gmodelpath, dmodelpath):
        self.generator.save(gmodelpath)
        self.discriminator.save(dmodelpath)
 
    def rgb_to_grayscale(self, rgb_image):
        r_coeff = 0.2989
        g_coeff = 0.5870
        b_coeff = 0.1140
        
        grayscale_image = np.dot(rgb_image[..., :3], [r_coeff, g_coeff, b_coeff])
        
        return grayscale_image

    def combine_patches(self, patches, IMGH=528, IMGW=528, num_h_crop=30, num_w_crop=30, loadSize=64, crop_stride=16):
        
        image = np.zeros((IMGH, IMGW, 3))
        patches_count = np.zeros((IMGH, IMGW, 3))
        patches_reshape = patches.reshape(num_h_crop, num_w_crop, loadSize, loadSize, 3)
        ps = loadSize  # crop patch size
        sd = crop_stride  # crop stride
        
        for idy in range(0, num_h_crop):
            crop_y = idy * sd
            if (idy * sd + ps) >= IMGH:
                crop_y = IMGH - ps
            for idx in range(0, num_w_crop):
                crop_x = idx * sd
                if (idx * sd + ps) >= IMGW:
                    crop_x = IMGW - ps
                image[crop_y:crop_y+ps, crop_x:crop_x+ps, :] += patches_reshape[idy][idx]
                patches_count[crop_y:crop_y+ps, crop_x:crop_x+ps, :] += 1.0

        image = image / patches_count
       
        image = self.rgb_to_grayscale(image)  # Assuming you have an 'rgb_to_grayscale' function defined
        
        return image

    def thresholding(self, image, top_k=0.02):
        num_pixels = image.flatten().shape[0]
        num_top_pixels = int(num_pixels * top_k)
        filter = np.partition(image.flatten(), -num_top_pixels)[-num_top_pixels]
        image[image>=filter] = 255
        image[image<filter] = 0
        return image

    def remove_small_areas(self, image, min_area=15):

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
        min_area_threshold = min_area
        
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
    
    def remove_padding(self, image):
        image = image[PADDING_PIXEL:-PADDING_PIXEL, PADDING_PIXEL:-PADDING_PIXEL]
        pad_width = ((EDGE_PIXEL, EDGE_PIXEL), (EDGE_PIXEL, EDGE_PIXEL))  # 上下左右各填充6个元素
        image = np.pad(image, pad_width, mode='constant', constant_values=0)
        return image

    def export_visualization(self, fn, mask):
        save_dir = os.path.join(opt.results_dir, 'img')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f'{fn}.png'), mask)

    def export_patches(self, fn, imgs, rec_imgs):
        save_dir = os.path.join(opt.results_dir, f'patches/{fn}/ori')
        os.makedirs(save_dir, exist_ok=True)
        imgs = ((imgs * 127.5) + 127.5).astype(np.uint8)
        for i, img in enumerate(imgs):
            img = Image.fromarray(img)
            img = ImageEnhance.Contrast(img).enhance(5)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'ori_{i}.png'), img)

        save_dir = os.path.join(opt.results_dir, f'patches/{fn}/rec')
        os.makedirs(save_dir, exist_ok=True)
        rec_imgs = ((rec_imgs * 127.5) + 127.5).astype(np.uint8)
        for i, img in enumerate(rec_imgs):
            img = Image.fromarray(img)
            img = ImageEnhance.Contrast(img).enhance(5)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'rec_{i}.png'), img)

    def visualize(self, patches):
        patches_combined = self.combine_patches(patches,)
        patches_threshold = self.thresholding(patches_combined,)
        patches_denoise = self.remove_small_areas(patches_threshold,)
        patches_result = self.remove_padding(patches_denoise)
        return patches_result

    def compute_mse(self, images, reconstructed_images):
        patches_diff = (images - reconstructed_images)**2
        # loss_rec = tf.math.square(images - reconstructed_images)
        
        patches_mse = patches_diff.mean()
        return patches_diff, patches_mse, patches_diff

    def generate_patches(self, images):
        images = tf.squeeze(images, axis=0)
        reconstructed_images = self.generator(images, training=False)
        images = images.numpy()
        reconstructed_images = reconstructed_images.numpy()
        return images, reconstructed_images
    
    def testing(self, test_dateset, g_filepath, d_filepath, save_dir):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)

        res_unsup = defaultdict(dict)
        for l in ['all', 'mean', 'label', 'fn']:
            for t in ['n','s']:
                res_unsup[l][t] = []
        n_count = 0
        s_count = 0
        for fp, images, labels in test_dateset: 
            if n_count >=2 and s_count >=2:
                break
            fn = fp.numpy()[0].decode('utf-8').split("/")[-1]
            
            images, reconstructed_images = self.generate_patches(images)
            _, mse_mean, all_mse = self.compute_mse(images, reconstructed_images)
            if labels == 0:
                res_unsup['all']['n'].extend(all_mse)
                res_unsup['fn']['n'].append(fn)
                res_unsup['mean']['n'].append(mse_mean)
                res_unsup['label']['n'].append(labels.numpy()[0])
                n_count+=1
                print(f"Normal {n_count}: {fn}")
            elif labels == 1:
                res_unsup['all']['s'].extend(all_mse)
                res_unsup['fn']['s'].extend(fn)
                res_unsup['mean']['s'].append(mse_mean)
                res_unsup['label']['s'].append(labels.numpy()[0])
                s_count+=1
                print(f"SMura {s_count}: {fn}")
            
        unsup_name = res_unsup['fn']['n'] + res_unsup['fn']['s']
        unsup_label = res_unsup['label']['n'] + res_unsup['label']['s']

        unsup_score_mean = [res_unsup['mean']['n'], res_unsup['mean']['s']]
        df_unsup_mean = pd.DataFrame(list(zip(unsup_name,unsup_score_mean,unsup_label)), columns=['name', 'score_mean', 'label'])
        df_unsup_mean.to_csv(os.path.join(save_dir, 'unsup_score_mean.csv'), index=False)     
        
        # res_unsup['all']['n'] = np.array(res_unsup['all']['n']).flatten()
        # res_unsup['all']['s'] = np.array(res_unsup['all']['s']).flatten()
        # score_all = np.append(res_unsup['all']['n'], res_unsup['all']['s'])
        # label_all = [0]*res_unsup['all']['n'].shape[0]+[1]*res_unsup['all']['s'].shape[0]
        # df_all = pd.DataFrame(list(zip(score_all,label_all)), columns=['score', 'label'])
        # df_all.to_csv(os.path.join(save_dir, 'unsup_score_all.csv'), index=False)
        # print("save score finished!")
    
def conv_block(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resunet_generator(input_shape):
    # print(inputs)
    # print("pretained start")
    """ Pre-trained ResNet50 Model """
    resnet50 = tf.keras.applications.ResNet50(include_top=True, weights="imagenet", input_tensor=input_shape)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (256 x 256)
    s2 = resnet50.get_layer("conv1_relu").output        ## (128 x 128)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (64 x 64)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (32 x 32)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (16 x 16)

    """ Decoder """
    x = 512
    d1 = decoder_block(b1, s4, x)                     ## (32 x 32)
    x = x/2
    d2 = decoder_block(d1, s3, x)                     ## (64 x 64)
    x = x/2
    d3 = decoder_block(d2, s2, x)                     ## (128 x 128)
    x = x/2
    d4 = decoder_block(d3, s1, x)                      ## (256 x 256)
    
    """ Output """
    # outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d4)
    outputs = tf.keras.layers.Conv2D(IMG_C, 1, padding="same", activation="tanh")(d4)
    # outputs = tf.keras.layers.Conv2D(3, 1, padding="same")(d5)

    model = tf.keras.models.Model(inputs, outputs)

    return model

def build_resunet_discriminator(inputs):
    num_layers = 4
    f = [2**i for i in range(num_layers)]
    x = inputs

    for i in range(0, num_layers):
        if i == 0:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * 512 ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        else:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * 512 ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            # x = tf.keras.layers.Dropout(0.3)(x)      
    
    x = tf.keras.layers.Flatten()(x)
    features = x
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.models.Model(inputs, outputs = [features, output])
    
    return model

class ResUnetGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator):
        super(ResUnetGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
       
        # Regularization Rate for each loss function
        self.ADV_REG_RATE_LF = 1
        self.REC_REG_RATE_LF = 50
        self.SSIM_REG_RATE_LF = 10
        self.FEAT_REG_RATE_LF = 1
        self.field_names = ['epoch', 'gen_loss', 'disc_loss']
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, beta_1=0.5, beta_2=0.999)
    
    def compile(self, g_optimizer, d_optimizer):
        super(ResUnetGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            reconstructed_images = self.generator(images, training=True)
            feature_real, label_real = self.discriminator(images, training=True)
            # print(generated_images.shape)
            feature_fake, label_fake = self.discriminator(reconstructed_images, training=True)
            # Loss 1: ADVERSARIAL loss
            
            real_loss = cross_entropy(label_real, tf.ones_like(label_real))
            fake_loss = cross_entropy(label_fake, tf.zeros_like(label_fake))
            disc_adv_loss = real_loss + fake_loss
            
            gen_adv_loss = cross_entropy(label_fake, tf.ones_like(label_real))
            
            # Loss 2: RECONSTRUCTION loss (L1)
            loss_rec = mae(images, reconstructed_images)
        
            # Loss 3: FEATURE Loss
            loss_feat = mse(feature_real, feature_fake)
            
            gen_loss = tf.reduce_mean( 
                (gen_adv_loss * self.ADV_REG_RATE_LF) 
                + (loss_rec * self.REC_REG_RATE_LF) 
                + (loss_feat * self.FEAT_REG_RATE_LF) 
            )
            
            disc_loss = tf.reduce_mean( (disc_adv_loss * self.ADV_REG_RATE_LF) + (loss_feat * self.FEAT_REG_RATE_LF) )


        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        

        
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        


        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "gen_adv_loss": gen_adv_loss,
            "disc_adv_loss": disc_adv_loss,
            "loss_rec": loss_rec,
            "loss_feat": loss_feat
        }

    def saved_model(self, gmodelpath, dmodelpath):
        self.generator.save(gmodelpath)
        self.discriminator.save(dmodelpath)
    
    def rgb_to_grayscale(self, rgb_image):
        r_coeff = 0.2989
        g_coeff = 0.5870
        b_coeff = 0.1140
        
        grayscale_image = np.dot(rgb_image[..., :3], [r_coeff, g_coeff, b_coeff])
        
        return grayscale_image

    def combine_patches(self, patches, IMGH=528, IMGW=528, num_h_crop=30, num_w_crop=30, loadSize=64, crop_stride=16):
        
        image = np.zeros((IMGH, IMGW, 3))
        patches_count = np.zeros((IMGH, IMGW, 3))
        patches_reshape = patches.reshape(num_h_crop, num_w_crop, loadSize, loadSize, 3)
        ps = loadSize  # crop patch size
        sd = crop_stride  # crop stride
        
        for idy in range(0, num_h_crop):
            crop_y = idy * sd
            if (idy * sd + ps) >= IMGH:
                crop_y = IMGH - ps
            for idx in range(0, num_w_crop):
                crop_x = idx * sd
                if (idx * sd + ps) >= IMGW:
                    crop_x = IMGW - ps
                image[crop_y:crop_y+ps, crop_x:crop_x+ps, :] += patches_reshape[idy][idx]
                patches_count[crop_y:crop_y+ps, crop_x:crop_x+ps, :] += 1.0

        image = image / patches_count
       
        image = self.rgb_to_grayscale(image)  # Assuming you have an 'rgb_to_grayscale' function defined
        
        return image

    def thresholding(self, image, top_k=0.02):
        num_pixels = image.flatten().shape[0]
        num_top_pixels = int(num_pixels * top_k)
        filter = np.partition(image.flatten(), -num_top_pixels)[-num_top_pixels]
        image[image>=filter] = 255
        image[image<filter] = 0
        return image

    def remove_small_areas(self, image, min_area=15):

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
        min_area_threshold = min_area
        
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
    
    def remove_padding(self, image):
        image = image[PADDING_PIXEL:-PADDING_PIXEL, PADDING_PIXEL:-PADDING_PIXEL]
        pad_width = ((EDGE_PIXEL, EDGE_PIXEL), (EDGE_PIXEL, EDGE_PIXEL))  # 上下左右各填充6个元素
        image = np.pad(image, pad_width, mode='constant', constant_values=0)
        return image

    def export_visualization(self, fn, mask):
        save_dir = os.path.join(opt.results_dir, 'img')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f'{fn}.png'), mask)

    def export_patches(self, fn, imgs, rec_imgs):
        save_dir = os.path.join(opt.results_dir, f'patches/{fn}/ori')
        os.makedirs(save_dir, exist_ok=True)
        imgs = ((imgs * 127.5) + 127.5).astype(np.uint8)
        for i, img in enumerate(imgs):
            img = Image.fromarray(img)
            img = ImageEnhance.Contrast(img).enhance(5)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'ori_{i}.png'), img)

        save_dir = os.path.join(opt.results_dir, f'patches/{fn}/rec')
        os.makedirs(save_dir, exist_ok=True)
        rec_imgs = ((rec_imgs * 127.5) + 127.5).astype(np.uint8)
        for i, img in enumerate(rec_imgs):
            img = Image.fromarray(img)
            img = ImageEnhance.Contrast(img).enhance(5)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'rec_{i}.png'), img)

    def visualize(self, patches):
        patches_combined = self.combine_patches(patches,)
        patches_threshold = self.thresholding(patches_combined,)
        patches_denoise = self.remove_small_areas(patches_threshold,)
        patches_result = self.remove_padding(patches_denoise)
        return patches_result

    def compute_mse(self, images, reconstructed_images):
        patches_diff = (images - reconstructed_images)**2
        # loss_rec = tf.math.square(images - reconstructed_images)
        
        patches_mse = patches_diff.mean()
        return patches_diff, patches_mse, patches_diff

    def generate_patches(self, images):
        images = tf.squeeze(images, axis=0)
        reconstructed_images = self.generator(images, training=False)
        images = images.numpy()
        reconstructed_images = reconstructed_images.numpy()
        return images, reconstructed_images
    
    def testing(self, test_dateset, g_filepath, d_filepath, save_dir):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)

        res_unsup = defaultdict(dict)
        for l in ['all', 'mean', 'label', 'fn']:
            for t in ['n','s']:
                res_unsup[l][t] = []
        n_count = 0
        s_count = 0
        for fp, images, labels in test_dateset: 
            
            fn = fp.numpy()[0].decode('utf-8').split("/")[-1]
            
            images, reconstructed_images = self.generate_patches(images)
            _, mse_mean, all_mse = self.compute_mse(images, reconstructed_images)
            if labels == 0:
                res_unsup['all']['n'].extend(all_mse)
                res_unsup['fn']['n'].append(fn)
                res_unsup['mean']['n'].append(mse_mean)
                res_unsup['label']['n'].append(labels.numpy()[0])
                n_count+=1
                print(f"Normal {n_count}: {fn}")
            elif labels == 1:
                print(fn)
                res_unsup['fn']['s'].extend(all_mse)
                res_unsup['mean']['s'].append(mse_mean)
                res_unsup['label']['s'].append(labels.numpy()[0])
                s_count+=1
                print(f"SMura {s_count}: {fn}")
            
        unsup_name = res_unsup['fn']['n'] + res_unsup['fn']['s']
        unsup_label = res_unsup['label']['n'] + res_unsup['label']['s']

        unsup_score_mean = [res_unsup['mean']['n'], res_unsup['mean']['s']]
        df_unsup_mean = pd.DataFrame(list(zip(unsup_name,unsup_score_mean,unsup_label)), columns=['name', 'score_mean', 'label'])
        df_unsup_mean.to_csv(os.path.join(save_dir, 'unsup_score_mean.csv'), index=False)     
        
        score_all = res_unsup['all']['n'] + res_unsup['all']['s']
        df_all = pd.DataFrame(list(zip(score_all,unsup_label)), columns=['score', 'label'])
        df_all.to_csv(os.path.join(save_dir, 'unsup_score_all.csv'), index=False)
        print("save score finished!")

# ===== training =====

def run_trainning(model, train_dataset, num_epochs, path_gmodal, path_dmodal, name_model):
    
    epochs_list = []
    gen_loss_list = []
    disc_loss_list = []
    l1_loss_list = []
    for epoch in range(0, num_epochs):
        epoch += 1
        print("running epoch: ", epoch)
        
        result = model.fit(
            train_dataset, 
            epochs = 1,
        )
        
        epochs_list.append(epoch)
        gen_loss_list.append(result.history["gen_loss"][0])
        disc_loss_list.append(result.history["disc_loss"][0])
        l1_loss_list.append(result.history["loss_rec"][0])
        model.saved_model(path_gmodal, path_dmodal)
        print('saved for epoch:', epoch)
    
    plot_epoch_result(epochs_list, gen_loss_list, "Generator_Loss", name_model, "g")
    plot_epoch_result(epochs_list, disc_loss_list, "Discriminator_Loss", name_model, "r")
    plot_epoch_result(epochs_list, l1_loss_list, "l1_Loss", name_model, "g")

# ===== other =====
def plot_epoch_result(epochs, loss, name, model_name, colour):
        plt.plot(epochs, loss, colour, label=name)
    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(model_name+ '_'+name+'_epoch_result.png')
        plt.show()
        plt.clf()

if __name__ == "__main__":
    print("start: ", opt.model_version)
    
    # set dir of files
    train_data_path = opt.dataroot
    saved_model_path = os.path.join(opt.checkpoints_dir, f"{opt.model_version}/saved_model/")
        
    path_gmodal = os.path.join(saved_model_path,f"{opt.model_version}_g_model.h5")
    path_dmodal = os.path.join(saved_model_path,f"{opt.model_version}_d_model.h5")
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    print(input_shape)
    inputs = tf.keras.layers.Input(input_shape, name="input_1")
    
    train_images = glob(train_data_path)    
    train_images_dataset = load_image_train(train_images, opt.batchSize)
    train_images_dataset = train_images_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=0.5, beta_2=0.999)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=0.5, beta_2=0.999)
    if "SkipGANomaly" in opt.model_version:
        g_model = build_skipgan_generator(inputs)
        d_model = build_skipgan_discriminator(inputs)
        model = SkipGanomaly(g_model, d_model)
        model.compile(g_optimizer, d_optimizer)
    elif "ResunetGAN" in opt.model_version:
        g_model = build_resunet_generator(inputs)
        d_model = build_resunet_discriminator(inputs)
        model = ResUnetGAN(g_model, d_model)
        model.compile(g_optimizer, d_optimizer)
        # resunetgan.testing(test_dateset, path_gmodal, path_dmodal, opt.results_dir)
    else:
        raise ValueError
    
    run_trainning(model, train_images_dataset, opt.niter, path_gmodal, path_dmodal, opt.model_version)






