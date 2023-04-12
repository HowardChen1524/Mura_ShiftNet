import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)
parser.add_argument('-os', '--overlap_strategy', type=str, default=None, required=True)
parser.add_argument('-ic', '--isCombine', action='store_true')

def join_path(p1,p2):
    return os.path.join(p1,p2)

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    overlap_strategy = args.overlap_strategy
    isCombine = args.isCombine

    exp_name_list = []
    hit_list = []
    dice_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    for exp in os.listdir(data_dir):
        exp_name_list.append(exp)
        with open(join_path(data_dir, f'{exp}/result.txt'), 'r') as f:
            for line in f:
                info = line.split(": ")
                if   info[0] == 'dice mean': dice_list.append(info[1][:-1])
                elif info[0] == 'recall': recall_list.append(info[1][:-1])
                elif info[0] == 'precision': precision_list.append(info[1][:-1])
                elif info[0] == 'f1-score': f1_list.append(info[1][:-1])
                elif info[0] == 'hit num': hit_list.append(info[1][:-1])
    df = pd.DataFrame(list(zip(exp_name_list, dice_list, recall_list, precision_list, f1_list, hit_list)), columns=['exp', 'dice mean', 'recall', 'precision', 'f1-score', 'hit num'])
    if isCombine:
        df[['grad_th', 'th', 'min_area']] = df['exp'].str.split('_', expand=True)[[1,3,4]].astype(float)

        # 按照数字和文本部分进行排序
        df_sorted = df.sort_values(['grad_th', 'th', 'min_area'])

        # 删除中间的列
        df_sorted = df_sorted.drop(['grad_th', 'th', 'min_area'], axis=1)
        
        df_sorted.to_csv(join_path(save_dir, f'combine_{overlap_strategy}_summary.csv'), index=False)
    else:
        df[['th', 'min_area']] = df['exp'].str.split('_', expand=True)[[0,4]].astype(float)

        # 按照数字和文本部分进行排序
        df_sorted = df.sort_values(['th', 'min_area'])

        # 删除中间的列
        df_sorted = df_sorted.drop(['th', 'min_area'], axis=1)

        df_sorted.to_csv(join_path(save_dir, f'unsup_{overlap_strategy}_summary.csv'), index=False)

            