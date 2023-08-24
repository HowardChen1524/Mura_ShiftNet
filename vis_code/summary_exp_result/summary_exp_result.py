import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-ct', '--combine_type', type=str, default=None, required=True)

def join_path(p1,p2):
    return os.path.join(p1,p2)

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    combine_type = args.combine_type

    exp_name_list = []
    hit_list = []
    dice_list = []
    recall_list = []
    precision_list = []

    for exp in os.listdir(data_dir):
        exp_name_list.append(exp)
        with open(join_path(data_dir, f'{exp}/result_all.txt'), 'r') as f:
            for line in f:
                info = line.split(": ")
                if   info[0] == 'dice mean': dice_list.append(info[1][:-1])
                elif info[0] == 'recall mean': recall_list.append(info[1][:-1])
                elif info[0] == 'precision mean': precision_list.append(info[1][:-1])
                elif info[0] == 'hit num': hit_list.append(info[1][:-1])
    df = pd.DataFrame(list(zip(exp_name_list, dice_list, recall_list, precision_list, hit_list)), columns=['exp', 'dice mean', 'recall', 'precision', 'hit num'])
    if combine_type == 'combine':
        df[['grad_th', 'th', 'min_area']] = df['exp'].str.split('_', expand=True)[[1,2,6]].astype(float)

        # 按照数字和文本部分进行排序
        df_sorted = df.sort_values(['grad_th', 'th', 'min_area'])

        # 删除中间的列
        df_sorted = df_sorted.drop(['grad_th', 'th', 'min_area'], axis=1)
        
        df_sorted.to_csv(join_path(data_dir, f'combine_summary.csv'), index=False)
    elif combine_type == 'unsup':
        df[['th', 'min_area']] = df['exp'].str.split('_', expand=True)[[0,4]].astype(float)

        # 按照数字和文本部分进行排序
        df_sorted = df.sort_values(['th', 'min_area'])

        # 删除中间的列
        df_sorted = df_sorted.drop(['th', 'min_area'], axis=1)

        df_sorted.to_csv(join_path(data_dir, f'unsup_summary.csv'), index=False)
    elif combine_type == 'sup':
        df[['grad_th']] = df['exp'].str.split('_', expand=True)[[1]].astype(float)

        # 按照数字和文本部分进行排序
        df_sorted = df.sort_values(['grad_th'])

        # 删除中间的列
        df_sorted = df_sorted.drop(['grad_th'], axis=1)

        df_sorted.to_csv(join_path(data_dir, f'sup_summary.csv'), index=False)

            