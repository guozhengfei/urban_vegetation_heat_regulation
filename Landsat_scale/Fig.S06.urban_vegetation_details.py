import numpy as np
import scipy.stats as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import os
import matplotlib.image as mpimg
import tifffile as tf
from PIL import Image

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

beijing_rgb_path = current_dir+'/1_Input/urban_vegetation_details/beijing_rgb.png'
img_rgb_bj = mpimg.imread(beijing_rgb_path)[:750,:1051,:]
beijing_lc_path = current_dir+'/1_Input/urban_vegetation_details/beijing_lc.png'
img_tif_bj = mpimg.imread(beijing_lc_path)
img = img_tif_bj[:,:,1]+img_tif_bj[:,:,2]
img = Image.fromarray(img)
# plt.figure();plt.imshow(img)
new_width = img_rgb_bj.shape[1]
new_height = img_rgb_bj.shape[0]
resized_img = img.resize((new_width, new_height))
resized_img = np.array(resized_img)
# plt.figure();plt.imshow(resized_img)
resized_img[resized_img<0.1]=np.nan
resized_img[resized_img>1]=np.nan
# resized_img[~np.isnan(resized_img)]=0.1

plt.figure(figsize=(8.5,7));plt.imshow(img_rgb_bj)
plt.imshow(resized_img,alpha=0.6,cmap='Greens_r')

figToPath = (current_dir + '/4_Figures/FigS06_Beijing_trees')
plt.savefig(figToPath, dpi=900)

london_rgb_path = current_dir+'/1_Input/urban_vegetation_details/london_rgb.png'
img_rgb_bj = mpimg.imread(london_rgb_path)
london_lc_path = current_dir+'/1_Input/urban_vegetation_details/london_lc.png'
img_tif_bj = mpimg.imread(london_lc_path)
img = img_tif_bj[:,:,1]+img_tif_bj[:,:,2]
img = Image.fromarray(img)
# plt.figure();plt.imshow(img)
new_width = img_rgb_bj.shape[1]
new_height = img_rgb_bj.shape[0]
resized_img = img.resize((new_width, new_height))
resized_img = np.array(resized_img)
# plt.figure();plt.imshow(resized_img)
resized_img[resized_img<0.1]=np.nan
resized_img[resized_img>1]=np.nan
# resized_img[~np.isnan(resized_img)]=0.1
plt.figure(figsize=(8.5,7));plt.imshow(img_rgb_bj)
plt.imshow(resized_img,alpha=0.6,cmap='Greens_r')

figToPath = (current_dir + '/4_Figures/FigS06_London_trees')
plt.savefig(figToPath, dpi=900)