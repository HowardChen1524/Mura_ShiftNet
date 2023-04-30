# Training
```bash
bash run_train.sh
```
### Parameters
<table style="float:center">
  <tr>
    <th><B> Parameters </B></th> <th><B> Description </B></th>
  </tr>
  <tr>
    <td>
    model_version
    </td>
    <td>
    模型訓練出來的版本名
    </td>
  </tr>
  <tr>
    <td>
    niter
    </td>
    <td>
    要訓練的 Epoch 數
    </td>
  </tr>
  <tr>
    <td>
    lr
    </td>
    <td>
    learning rate
    </td>
  </tr> 
  <tr>
    <td>
    lr_policy
    </td>
    <td>
    Scheduler
    </td>
  </tr> 
  <tr>
    <td>
    crop_image_num
    </td>
    <td>
    訓練時每張 image 切出的 patch 數量
    </td>
  </tr> 
  <tr>
    <td>
    loadSize
    </td>
    <td>
    每個 patch 的大小
    </td>
  </tr> 
  <tr>
    <td>
    resolution
    </td>
    <td>
    圖片訓練的解析度 [origin | resized]
    </td>
  </tr> 
  <tr>
    <td>
    train_normal_path
    </td>
    <td>
    訓練資料的路徑
    </td>
  </tr> 
  <tr>
    <td>
    checkpoints_dir
    </td>
    <td>
    訓練時模型保存的路徑
    </td>
  </tr> 
</table>

# Testing 
## Mura Binary classification
```bash
bash run_test.sh
```

## Find serious Mura Location
```bash
bash run_gen_loc_union.sh # for union method
bash run_gen_loc_average.sh # for average method
```

### Parameters
<table style="float:center">
  <tr>
    <th><B> Parameters </B></th> <th><B> Description </B></th>
  </tr>
  <tr>
    <td>
    model_version
    </td>
    <td>
    模型訓練出來的版本名
    </td>
  </tr>
  <tr>
    <td>
    which_epoch
    </td>
    <td>
    使用 model_version 中的第幾個 epoch 的模型
    </td>
  </tr>
  <tr>
    <td>
    crop_stride
    </td>
    <td>
    將 image 做 sliding crop 時的 stride
    </td>
  </tr> 
  <tr>
    <td>
    measure_mode
    </td>
    <td>
    計算 Anomaly score 的方式 [MSE | Content | Feat]
    </td>
  </tr> 
  <tr>
    <td>
    dataset_version
    </td>
    <td>
    資料集的版本
    </td>
  </tr> 
  <tr>
    <td>
    loadSize
    </td>
    <td>
    每個 patch 的大小
    </td>
  </tr> 
  <tr>
    <td>
    resolution
    </td>
    <td>
    圖片訓練的解析度 [origin | resized]
    </td>
  </tr> 
  <tr>
    <td>
    train_normal_path
    </td>
    <td>
    訓練資料的路徑
    </td>
  </tr> 
  <tr>
    <td>
    checkpoints_dir
    </td>
    <td>
    訓練時模型保存的路徑
    </td>
  </tr> 
  <tr>
    <td>
    mask_part
    </td>
    <td>
    是否只使用挖空處計算 Anomaly score
    </td>
  </tr>
  <tr>
    <td>
    pos_normalize
    </td>
    <td>
    是否使用位置標準化
    </td>
  </tr>
  <tr>
    <td>
    overlap_strategy
    </td>
    <td>
    視覺化差異的方式 [union | average]
    </td>
  </tr>
  <tr>
    <td>
    binary_threshold
    </td>
    <td>
    視覺化差異時的二值化 threshold (only for union method)
    </td>
  </tr>
  <tr>
    <td>
    top_k
    </td>
    <td>
    視覺化差異時，取差異值在 top_k 的值為 threshold (only for average method)
    </td>
  </tr>
  <tr>
    <td>
    isPadding
    </td>
    <td>
    是否使用 padding (一般只有在有做 resize 時才會使用)
    </td>
  </tr>
  <tr>
    <td>
    isResize
    </td>
    <td>
    是否進行 resize (origin: 1920*1080, resized:512*512)
    </td>
  </tr>
  <tr>
    <td>
    sup_model_version
    </td>
    <td>
    supervised 模型的版本名
    </td>
  </tr>
  <tr>
    <td>
    sup_gradcam_th
    </td>
    <td>
    gradcam 的 threshold
    </td>
  </tr>

</table>


