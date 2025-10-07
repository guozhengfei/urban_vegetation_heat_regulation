import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.ndimage import convolve1d
import tifffile as tf
import cv2
import matplotlib; matplotlib.use('Qt5Agg')


def aggregate_array(arr, factor):
    new_shape = (arr.shape[0] // factor, arr.shape[1] // factor)
    arr_cropped = arr[:new_shape[0]*factor, :new_shape[1]*factor]
    arr_reshaped = arr_cropped.reshape(new_shape[0], factor, new_shape[1], factor)
    arr_agg = arr_reshaped.mean(axis=(1, 3))
    return arr_agg

def load_and_resize(path, shape):
    arr = tf.imread(path)
    arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return arr

current_dir = os.path.dirname(os.path.dirname(os.getcwd()).replace('\\', '/'))
cities = [1027.0,858.0,807.0,1080.0,867.0,936.0,992.0,785.0,767.0]  # 替换为你的城市名列表
factors = [1, 3, 16, 33]
labels = ['tree', 'grass', 'crop', 'build']

all_pred_pures = []

for city in cities:
    # 假设每个城市有独立的文件夹，路径可根据实际情况调整
    city_folder = f"{current_dir}/1_Input/sensitivity/"
    et_path = city_folder + f'ET_2020_30m_{city}.tif'
    with rasterio.open(et_path) as src:
        arr = src.read(1)
        arr_shape = arr.shape

    treeC = load_and_resize(current_dir + '/1_Input/treeCover_100m/' + f'cropC_{city}.tif', arr_shape)
    grassC = load_and_resize(current_dir + '/1_Input/grassCover_100m/' + f'grassC_{city}.tif', arr_shape)
    cropC = load_and_resize(current_dir + '/1_Input/cropCover_100m/' + f'cropC_{city}.tif', arr_shape)
    buildC = load_and_resize(current_dir + '/1_Input/buildCover_100m/' + f'buildC_{city}.tif', arr_shape)
    waterC = load_and_resize(current_dir + '/1_Input/waterCover_100m/' + f'waterC_{city}.tif', arr_shape)

    cover_dict = {
        'treeC': treeC,
        'grassC': grassC,
        'cropC': cropC,
        'buildC': buildC,
        'waterC': waterC
    }

    pred_pures = []
    for idx, f in enumerate(factors):
        agg_cover = {name: aggregate_array(arr_, f) for name, arr_ in cover_dict.items()}
        agg_et = aggregate_array(arr, f)

        X = np.stack([agg_cover['treeC'].flatten(),
                      agg_cover['grassC'].flatten(),
                      agg_cover['cropC'].flatten(),
                      agg_cover['buildC'].flatten(),
                      agg_cover['waterC'].flatten()], axis=1)
        y = agg_et.flatten()

        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        model = xgb.XGBRegressor(max_depth=2, n_estimators=100, learning_rate=0.07, tree_method='hist')
        model.fit(X, y)
        pred_pure = model.predict([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
        pred_pures.append(pred_pure)
    pred_pures_arr = np.array(pred_pures)
    all_pred_pures.append(pred_pures_arr)

all_pred_pures = np.array(all_pred_pures)  # shape: (n_cities, n_res, n_types)

# 以每个城市的第一行为reference，计算相对bias并画图

pred_pures_mean = np.nanmean(all_pred_pures,axis=0)
ref = pred_pures_mean[0]
relative_bias = abs(pred_pures_mean - ref) / ref * 100  # 百分比
sd = np.nanstd(all_pred_pures,axis=0)*0.25

plt.figure(figsize=(5*0.8, 3.5*0.8))
x = np.arange(pred_pures_mean.shape[0])
for i in range(pred_pures_mean.shape[1]):
    plt.errorbar(x, pred_pures_mean[:, i], yerr=sd[:, i], marker='o', capsize=4, label=labels[i])
plt.xlabel('Resolution')
plt.xticks(range(4), ['30m', '90m', '510m', '990m'])
plt.ylabel('Predicted Pure ET')
plt.tight_layout()
outpath = current_dir + f'/4_Figures/ET_along_resolution.png'
plt.savefig(outpath, dpi=600)
plt.close()

plt.figure(figsize=(5*0.8, 3.5*0.8))
sd2 = sd/pred_pures_mean*10
for i in range(relative_bias.shape[1]):
    plt.errorbar(range(4), relative_bias[:, i], yerr=sd2[:, i], marker='o', label=labels[i])
plt.ylabel('Relative Bias (%)')
plt.xlabel('Resolution')
plt.xticks(range(4), ['30m', '90m', '510m', '990m'])
plt.legend()
plt.tight_layout()
outpath = current_dir + f'/4_Figures/ET_along_resolution_bias.png'
plt.savefig(outpath, dpi=600)
