import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import tifffile as tf
from matplotlib.patches import Circle

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
# warmer grass case
lst_path = current_dir+'/1_Input/real cases/Landsat30m_grass_warmer.tif'

lst = tf.imread(lst_path)
lst[:,:,0] = (lst[:,:,-1]+lst[:,:,1])*0.5
lst[:,:,2] = (lst[:,:,1]+lst[:,:,3])*0.5

plt.figure();plt.imshow(np.nanmean(lst,axis=2)[:23,:23],cmap='RdYlBu_r')
fig = plt.figure(figsize=(12,9))
for i in range(1,12+1):
    ax = fig.add_subplot(3, 4, i)
    plt.imshow(lst[:23,:23,i-1],cmap='RdYlBu_r')
    circ = Circle((11, 11), 1,fill=True, color='g')
    ax.add_patch(circ)
    plt.tick_params(labelsize=0)

figToPath = r'E:\project 1 urban vegetation thermoregulation\4_Figures\warmer_grass'
plt.savefig(figToPath, dpi=600)
plt.close()

t_grass = lst[4:13, 7:16, :]
t_build = lst*1
t_build[4:13, 7:16, :] = np.nan
plt.figure(figsize=(4.7,3))
t_grass = t_grass.reshape(t_grass.shape[0]*t_grass.shape[1],t_grass.shape[2])
t_grass_mean = np.nanmean(t_grass,axis=0)
t_grass_sd = np.nanstd(t_grass,axis=0)*0.5
plt.plot(np.linspace(1,12,12),t_grass_mean,color='blue',lw=2)
plt.fill_between(np.linspace(1,12,12), t_grass_mean - t_grass_sd, t_grass_mean + t_grass_sd, color='blue', alpha=0.3, label='Standard Deviation')

t_build = t_build.reshape(t_build.shape[0]*t_build.shape[1],t_build.shape[2])
t_build_mean = np.nanmean(t_build,axis=0)
t_build_sd = np.nanstd(t_build,axis=0)*0.5
plt.plot(np.linspace(1,12,12),t_build_mean,color='red',lw=2)
plt.fill_between(np.linspace(1,12,12), t_build_mean - t_build_sd, t_build_mean + t_build_sd, color='red', alpha=0.3, label='Standard Deviation')
plt.tick_params(labelsize=12)
figToPath = current_dir+'/4_Figures/warmer_grass_line'
plt.savefig(figToPath, dpi=600)
plt.close()

# warmer cropland case
lst_path2 = current_dir+'/1_Input/real cases/Landsat30m_cropland_warmer.tif'
lst = tf.imread(lst_path2)
plt.figure();plt.imshow(np.nanmean(lst,axis=2)[:,:23],cmap='RdYlBu_r')

fig = plt.figure(figsize=(12,9))
for i in range(1,12+1):
    ax = fig.add_subplot(3, 4, i)
    plt.imshow(lst[:,:23,i-1],cmap='RdYlBu_r')
    circ = Circle((11, 8), 1,fill=True, color='g')
    ax.add_patch(circ)
    plt.tick_params(labelsize=0)

figToPath = r'E:\project 1 urban vegetation thermoregulation\4_Figures\warmer_cropland'
plt.savefig(figToPath, dpi=600)
plt.close()

t_crop = lst[:, 14:, :]
t_build = lst*1
t_build[:, 14:, :] = np.nan

plt.figure(figsize=(4.7,3))
t_crop = t_crop.reshape(t_crop.shape[0]*t_crop.shape[1],t_crop.shape[2])
t_crop_mean = np.nanmean(t_crop,axis=0)
t_crop_sd = np.nanstd(t_crop,axis=0)*0.5
plt.plot(np.linspace(1,12,12),t_crop_mean,color='blue',lw=2)
plt.fill_between(np.linspace(1,12,12), t_crop_mean - t_crop_sd, t_crop_mean + t_crop_sd, color='blue', alpha=0.3, label='Standard Deviation')

t_build = t_build.reshape(t_build.shape[0]*t_build.shape[1],t_build.shape[2])
t_build_mean = np.nanmean(t_build,axis=0)
t_build_sd = np.nanstd(t_build,axis=0)*0.5
plt.plot(np.linspace(1,12,12),t_build_mean,color='red',lw=2)
plt.fill_between(np.linspace(1,12,12), t_build_mean - t_build_sd, t_build_mean + t_build_sd, color='red', alpha=0.3, label='Standard Deviation')
plt.tick_params(labelsize=12)
figToPath = current_dir+'/4_Figures/warmer_crop_line'
plt.savefig(figToPath, dpi=600)
plt.close()

# warmer tree case
lst_path3 = current_dir+'/1_Input/real cases/Landsat30m_tree_warmer.tif'
lst = tf.imread(lst_path3)
lst_mean = np.nanmean(lst,axis=2)
lst[:,:,2] = (lst[:,:,1]+lst[:,:,3])*0.5
lst[:,:,8] = np.nanmean(lst[:,:,8]-lst_mean)+lst_mean
lst[:,:,9] = np.nanmean(lst[:,:,8]-lst_mean)+lst_mean
lst[:,:,10] = (lst[:,:,9]+lst[:,:,11])*0.5
plt.figure();plt.imshow(np.nanmean(lst,axis=2)[:,:23],cmap='RdYlBu_r')

fig = plt.figure(figsize=(12,9))
for i in range(1,12+1):
    ax = fig.add_subplot(3, 4, i)
    plt.imshow(lst[:,:,i-1],cmap='RdYlBu_r')
    circ = Circle((15, 11), 1, fill=True, color='g')
    ax.add_patch(circ)
    plt.tick_params(labelsize=0)

figToPath = r'E:\project 1 urban vegetation thermoregulation\4_Figures\warmer_cropland'
plt.savefig(figToPath, dpi=600)
plt.close()

t_tree = lst[:, 3:13, :]
t_build = lst*1
t_build[:, 3:13, :] = np.nan

plt.figure(figsize=(4.7,3))
t_tree = t_tree.reshape(t_tree.shape[0]*t_tree.shape[1],t_tree.shape[2])
t_tree_mean = np.nanmean(t_tree,axis=0)
t_tree_sd = np.nanstd(t_tree,axis=0)*0.5
plt.plot(np.linspace(1,12,12),t_tree_mean,color='blue',lw=2)
plt.fill_between(np.linspace(1,12,12), t_tree_mean - t_tree_sd, t_tree_mean + t_tree_sd, color='blue', alpha=0.3, label='Standard Deviation')

t_build = t_build.reshape(t_build.shape[0]*t_build.shape[1],t_build.shape[2])
t_build_mean = np.nanmean(t_build,axis=0)
t_build_sd = np.nanstd(t_build,axis=0)*0.5
plt.plot(np.linspace(1,12,12),t_build_mean,color='red',lw=2)
plt.fill_between(np.linspace(1,12,12), t_build_mean - t_build_sd, t_build_mean + t_build_sd, color='red', alpha=0.3, label='Standard Deviation')
plt.tick_params(labelsize=12)
figToPath = current_dir+'/4_Figures/warmer_tree_line'
plt.savefig(figToPath, dpi=600)
plt.close()

# cooler tree case
lst_path4 = r'E:\project 1 urban vegetation thermoregulation\1_Input\real cases\Landsat30m_tree_cooler.tif'
lst = tf.imread(lst_path4)
lst_mean = np.nanmean(lst,axis=2)
plt.figure();plt.imshow(np.nanmean(lst,axis=2),cmap='RdYlBu_r')

# cooler grass case
lst_path5 = r'E:\project 1 urban vegetation thermoregulation\1_Input\real cases\Landsat30m_grass_cooler.tif'
lst = tf.imread(lst_path5)
lst_mean = np.nanmean(lst,axis=2)
plt.figure();plt.imshow(np.nanmean(lst,axis=2),cmap='RdYlBu_r')

# cooler cropland case
lst_path6 = r'E:\project 1 urban vegetation thermoregulation\1_Input\real cases\Landsat30m_cropland_cooler.tif'
lst = tf.imread(lst_path6)
lst_mean = np.nanmean(lst,axis=2)
plt.figure();plt.imshow(np.nanmean(lst,axis=2),cmap='RdYlBu_r')