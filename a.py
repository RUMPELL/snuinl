import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import os
import shutil

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
print("Using PyTorch version:", torch.__version__, 'Device: ', DEVICE)
    


path2xls = './IXI.xls'
labels_df = pd.read_excel(path2xls)
labels_df.head()

labels_df['IXI_ID']

age_df = labels_df[['IXI_ID', 'AGE']]
age_df


path_dir = './IXI'
file_list = os.listdir(path_dir)



int(file_list[0][8:11])

def indexing(x):
    return int(x[8:11])
    

f_list=[]
for f in file_list:
    x = indexing(f)
    f_list.append(x)
    



len(f_list)
IXI_df = age_df[age_df['IXI_ID'].isin(f_list)]