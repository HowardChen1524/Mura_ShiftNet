import cv2
import numpy as np

# 讀取二值化影像
img = cv2.imread('/home/sallylab/Howard/Mura_ShiftNet/detect_position/typed/16/union/0.0125_diff_pos_area_0/imgs/7A2DBCJ9KBZZ_044619_0_L050P_Img_OriginalROI_WithoutDraw.png', cv2.IMREAD_GRAYSCALE)

# 使用 connectedComponents 函數
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

# 輸出連通區域的數量
print("連通區域的數量：", num_labels)

# 輸出每個區域的統計資訊
for i in range(1, num_labels):
    left = stats[i, cv2.CC_STAT_LEFT]
    top = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    print("區域", i, "的左上角座標：", (left, top), "寬度：", width, "高度：", height, "面積：", area)

# 將每個區域的標籤轉換為彩色影像並顯示
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0
cv2.imwrite('labeled_image.png', labeled_img)

# 指定面積閾值
max_area_threshold = 200
min_area_threshold = 2
# 遍歷所有區域
for i in range(1, num_labels):
    # 如果區域面積小於閾值，就將對應的像素值設置為黑色
    if stats[i, cv2.CC_STAT_AREA] < min_area_threshold or stats[i, cv2.CC_STAT_AREA] > max_area_threshold:
        labels[labels == i] = 0

# 將標籤為 0 的像素設置為白色，其它像素設置為黑色
result = labels.astype('uint8')
print(np.unique(labels))
result[result == 0] = 0
result[result != 0] = 255
cv2.imwrite('result.png', result)

# result[result == 0] = 255
# result[result != 255] = 0

# 顯示結果
cv2.imwrite('result.png', result)


