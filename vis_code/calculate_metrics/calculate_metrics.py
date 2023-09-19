import os
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-sgd', '--seg_gt_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)
parser.add_argument('-ir', '--isResize', type=int, default=None, required=True)
parser.add_argument('-cd', '--csv_dir', type=str, default=None, required=True)
parser.add_argument('-bgd', '--bb_gt_dir', type=str, default=None, required=True)

def compute_recall_precision(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Error: Images have different shapes"
    tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    return recall, precision

def dice_coefficient(img1, img2):    
    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "Error: Images have different shapes"
    # Calculate the Dice coefficient
    # Calculate the intersection
    intersection = np.sum(img1 * img2)
    total_white_pixel = np.sum(img1) + np.sum(img2)

    dice = (2 * intersection) / total_white_pixel
    return dice

def join_path(p1,p2):
    return os.path.join(p1,p2)

def area_based_metric(gt_dir, data_dir, save_dir):
    dice_mean = defaultdict(float)
    recall_mean = defaultdict(float)
    precision_mean = defaultdict(float)
    count = 0
    for fn in os.listdir(gt_dir):
        count += 1   
        # Load the images
        thresh = 127
        gt_img = cv2.imread(join_path(gt_dir,fn), cv2.IMREAD_GRAYSCALE)
        diff_img = cv2.imread(join_path(data_dir,fn), cv2.IMREAD_GRAYSCALE)
        if isResize == 1:
            gt_img = cv2.resize(gt_img, (512,512), interpolation=cv2.INTER_LINEAR)
            diff_img = cv2.resize(diff_img, (512,512), interpolation=cv2.INTER_LINEAR)
        gt_img = cv2.threshold(gt_img, thresh, 255, cv2.THRESH_BINARY)[1]/255
        diff_img = cv2.threshold(diff_img, thresh, 255, cv2.THRESH_BINARY)[1]/255

        dice = dice_coefficient(gt_img, diff_img)
        dice_mean[fn] = dice            
        recall, precision = compute_recall_precision(gt_img, diff_img)
        recall_mean[fn] = recall
        precision_mean[fn] = precision
        print("Num {}: {}\ndice: {}, recall: {}, precision:{}".format(count, fn, dice, recall, precision))
        
    df_dice = pd.DataFrame(data=list(dice_mean.items()),columns=['fn','dice'])
    df_recall = pd.DataFrame(data=list(recall_mean.items()),columns=['fn','recall'])
    df_precision = pd.DataFrame(data=list(precision_mean.items()),columns=['fn','precision'])

    print(f"finished, dice mean:{df_dice['dice'].mean()}")
    print(f"finished, recall mean:{df_recall['recall'].mean()}")
    print(f"finished, precision mean:{df_precision['precision'].mean()}")
    df_dice.to_csv(join_path(save_dir, f'dice_all.csv'),index=False)
    df_recall.to_csv(join_path(save_dir, f'recall_all.csv'),index=False)
    df_precision.to_csv(join_path(save_dir, f'precision_all.csv'),index=False)

    with open(join_path(save_dir, f"area_based_result_all.txt"), 'w') as f:
        msg = f"All img: {count}\n" 
        msg += f"hit num: {df_dice[df_dice['dice']>0].shape[0]}\n"
        msg += f"dice mean: {df_dice['dice'].mean()}\n"
        msg += f"recall mean: {df_recall['recall'].mean()}\n"
        msg += f"precision mean: {df_precision['precision'].mean()}\n"
        f.writelines(msg)

def calculate_iou(box1, box2):
    # Calculate the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of the two boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def count_successful_hits(ground_truth_boxes, predicted_boxes, threshold):
    hits = 0

    for gt_box in ground_truth_boxes:
        for pred_box in predicted_boxes:
            iou = calculate_iou(gt_box, pred_box)
            if iou >= threshold:
                hits += 1
                break  # Break the loop to avoid double counting hits

    return hits

def defect_based_metric(csv_dir, data_dir, save_dir, iou_th=0.3):
    total_smura_num = 0
    total_pred_num = 0
    total_hit_num = 0
    df = pd.read_csv(csv_dir)
    count = 0
    for fn in os.listdir(data_dir):
        count += 1 
        # Load the images
        thresh = 127
        diff_img = cv2.imread(join_path(data_dir,fn), cv2.IMREAD_GRAYSCALE)
        if isResize == 1:
            diff_img = cv2.resize(diff_img, (512,512), interpolation=cv2.INTER_LINEAR)
        diff_img = cv2.threshold(diff_img, thresh, 255, cv2.THRESH_BINARY)[1]

        # Find contours in the binary mask
        contours, _ = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # Filter and extract bounding boxes
        min_contour_area = 1 # 最小輪廓面積，因在生成圖片時已經過濾最小面積，此處設1即可
        predicted_boxes = []

        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                # csv
                total_pred_num+=1
                predicted_boxes.append((x, y, x + w, y + h))

        # Print the predicted bounding box coordinates
        # print(predicted_boxes)
        ground_truth_boxes = []
        gt_df = df[df['fn']==fn]
        for row in gt_df.itertuples():
            total_smura_num+=1
            ground_truth_boxes.append((int(round(row.x0/3.75,0)), int(round(row.y0/2.109375,0)), int(round(row.x1/3.75,0)), int(round(row.y1/2.109375,0))))
        # print(ground_truth_boxes)

        threshold = iou_th

        hit_count = count_successful_hits(ground_truth_boxes, predicted_boxes, threshold)
        print("Num {}: {}".format(count, fn))

        total_hit_num += hit_count
    
    with open(join_path(save_dir, f"defect_based_iou_{iou_th}.txt"), 'w') as f:
        msg = f"All img: {count}\n" 
        msg += f"IOU th: {iou_th}\n" 
        msg += f"Number of gt: {total_smura_num}\n"
        msg += f"Number of pred: {total_pred_num}\n"        
        msg += f"Number of hits: {total_hit_num}\n"
        msg += f"Recall: {total_hit_num/total_smura_num}\n"
        msg += f"Precision: {total_hit_num/total_pred_num}\n"


        f.writelines(msg)

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    seg_gt_dir = args.seg_gt_dir
    save_dir = args.save_dir
    isResize = args.isResize
    csv_dir = args.csv_dir
    bb_gt_dir = args.bb_gt_dir
    os.makedirs(save_dir, exist_ok=True)
    area_based_metric(seg_gt_dir, data_dir, save_dir)
    defect_based_metric(csv_dir, data_dir, save_dir)