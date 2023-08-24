from glob import glob
import pandas as pd
from collections import defaultdict 
import xmltodict, json
import argparse
import os

def add_row(obj, dw=1920, dh=1080):
    info_fn = defaultdict()

    # class
    if obj['name'] == '黑塊':
        info_fn['class'] = 0
    elif obj['name'] == '白塊':
        info_fn['class'] = 1
    
    # pos
    x0 = int(obj['bndbox']['xmin'])
    y0 = int(obj['bndbox']['ymin'])
    x1 = int(obj['bndbox']['xmax'])
    y1 = int(obj['bndbox']['ymax'])
    info_fn['x_center'] = ((x0+x1)/2)/dw
    info_fn['y_center'] = ((y0+y1)/2)/dh
    info_fn['w'] = (x1-x0)/dw
    info_fn['h'] = (y1-y0)/dh
    
    return info_fn

parser = argparse.ArgumentParser()
parser.add_argument('-xd', '--xml_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

args = parser.parse_args()
xml_dir = args.xml_dir
save_dir = args.save_dir

xml_list = glob(f"{xml_dir}*xml")
for xml_path in xml_list:
    info_fn_list = []
    with open(xml_path) as fd:
        json_fd = xmltodict.parse(fd.read())
        if isinstance(json_fd['annotation']['object'], list):
            for obj in json_fd['annotation']['object']:
                info_fn_list.append(add_row(obj))
        else:
            info_fn_list.append(add_row(json_fd['annotation']['object']))

        df = pd.DataFrame.from_dict(info_fn_list)
        df.to_csv(os.path.join(save_dir, f"{json_fd['annotation']['filename'][:-4]}.txt"), sep=' ', index=False, header=False)