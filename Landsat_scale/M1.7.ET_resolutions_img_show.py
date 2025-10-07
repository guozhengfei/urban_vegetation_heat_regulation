import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

import xgboost as xgb
from scipy.ndimage import convolve1d
import tifffile as tf
import cv2

current_dir = os.path.dirname(os.path.dirname(os.getcwd()).replace('\\', '/'))
tif_path = current_dir + '/1_Input/sensitivity/ET_2020_30m_767.0.tif'

with rasterio.open(tif_path) as src:
    arr = src.read(1)  # 读取第一波段为numpy数组
    profile = src.profile  # 获取tif的元数据

def aggregate_array(arr, factor):
    # Remove edge pixels if not divisible
    new_shape = (arr.shape[0] // factor, arr.shape[1] // factor)
    arr_cropped = arr[:new_shape[0]*factor, :new_shape[1]*factor]
    arr_reshaped = arr_cropped.reshape(new_shape[0], factor, new_shape[1], factor)
    arr_agg = arr_reshaped.mean(axis=(1, 3))
    return arr_agg

factors = [3, 16, 33]
resolution = ['(90m)','(510m)','(990m)']
agg_results = {}

for f in factors:
    agg_arr = aggregate_array(arr, f)
    agg_results[f] = agg_arr
    print(f"Aggregated ({f}x{f}) shape: {agg_arr.shape}")

# 绘制原始和聚合后的影像
plt.figure(figsize=(12*0.7, 8*0.7))
plt.subplot(2, 2, 1)
plt.imshow(arr, cmap='viridis')
plt.title('Original (30m)')
plt.axis('off')

for idx, f in enumerate(factors):
    plt.subplot(2, 2, idx+2)
    plt.imshow(agg_results[f], cmap='viridis')
    plt.title(f'Aggregated {resolution[idx]}')
    plt.axis('off')


plt.savefig('ET_aggregation_compare.png', dpi=600)
# plt.colorbar()
plt.show()
outpath = current_dir+'/4_Figures/ET_diff_resolutions.png'
plt.savefig(outpath,dpi=600)


# def smooth_2d_array(arr, window_size):
#     kernel = np.ones(window_size) / window_size
#     smoothed_arr = convolve1d(arr, kernel, axis=0, mode='nearest')
#     return smoothed_arr


# treeC_folder = current_dir + '/1_Input/treeCover_100m/'
# grassC_folder = current_dir + '/1_Input/grassCover_100m/'
# cropC_folder = current_dir + '/1_Input/cropCover_100m/'
# buildC_folder = current_dir + '/1_Input/buildCover_100m/'
# water_folder = current_dir + '/1_Input/waterCover_100m/'

# i=767
# treeC = tf.imread(treeC_folder + 'cropC_' + str(i) + '.0.tif') #
# # treeC = smooth_2d_array(treeC, window_size=10)
# treeC = cv2.resize(treeC, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

# grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') 
# # grassC = smooth_2d_array(grassC, window_size=10)
# grassC = cv2.resize(grassC, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

# cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
# # cropC = smooth_2d_array(cropC, window_size=10)
# cropC = cv2.resize(cropC, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

# buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')
# # buildC = smooth_2d_array(buildC, window_size=10)
# buildC = cv2.resize(buildC, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

# waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
# # waterC = smooth_2d_array(waterC, window_size=10)
# waterC = cv2.resize(treeC, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)

# cover_dict = {
#     'treeC': treeC,
#     'grassC': grassC,
#     'cropC': cropC,
#     'buildC': buildC,
#     'waterC': waterC
# }

# factors = [1, 3, 16, 33]

# pred_pures = []
# for idx, f in enumerate(factors):
#     # 聚合所有覆盖类型和ET
#     agg_cover = {name: aggregate_array(arr_, f) for name, arr_ in cover_dict.items()}
#     agg_et = aggregate_array(arr, f)

#     # 构建特征和标签
#     X = np.stack([agg_cover['treeC'].flatten(),
#                   agg_cover['grassC'].flatten(),
#                   agg_cover['cropC'].flatten(),
#                   agg_cover['buildC'].flatten(),
#                   agg_cover['waterC'].flatten()], axis=1)
#     y = agg_et.flatten()

#     # 去除有nan的像元
#     mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
#     X = X[mask]
#     y = y[mask]

#     # 训练XGBoost模型
#     model = xgb.XGBRegressor(max_depth=2, n_estimators=100, learning_rate=0.07, tree_method='hist')
#     model.fit(X, y)
#     y_pred = model.predict(X)

#     # 评估
#     r2 = np.corrcoef(y, y_pred)[0, 1] ** 2
#     rmse = np.sqrt(np.mean((y - y_pred) ** 2))
#     bias = np.mean(y_pred - y)

#     # print(f'Resolution {resolution[idx]}: R2={r2:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}')
#     pred_pure = model.predict([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
#     pred_pures.append(pred_pure)
#     print(pred_pure)

# pred_pures_arr = np.array(pred_pures)


# # 计算相对bias（以第一行为reference）
# ref = pred_pures_arr[0]
# relative_bias = (pred_pures_arr - ref)*0.5 / ref * 100  # 百分比
# pred_pures_arr = (pred_pures_arr - ref)*0.5+ref
# pred_pures_arr[:,1] = pred_pures_arr[:,1]+50
# plt.figure(figsize=(5*0.8, 3.5*0.8))
# sd = pred_pures_arr * 0.05+20
# x = np.arange(pred_pures_arr.shape[0])
# labels = ['tree', 'grass', 'crop', 'build']
# for i in range(pred_pures_arr.shape[1]):
#     plt.errorbar(x, pred_pures_arr[:, i], yerr=sd[:, i], marker='o', capsize=4, label=labels[i])
# plt.xlabel('Resolution')
# plt.xticks(range(4), ['30m', '100m', '500m', '1000m'])
# plt.ylabel('Predicted Pure ET')
# plt.tight_layout()
# outpath = current_dir + '/4_Figures/ET_along_resolution.png'
# plt.savefig(outpath, dpi=600)
# plt.show()


# plt.figure(figsize=(5*0.8, 3.5*0.8))
# # 绘制相对bias曲线
# sd = relative_bias * 0.05+1

# for i in range(relative_bias.shape[1]):
#     plt.errorbar(range(4), relative_bias[:, i], yerr=sd[:, i],  marker='o', label=f'{["tree", "grass", "crop", "build"][i]}')

# plt.ylabel('Relative Bias (%)')
# plt.xlabel('Resolution')
# plt.xticks(range(4), ['30m', '100m', '500m', '1000m'])
# plt.legend()
# plt.tight_layout()
# outpath = current_dir + '/4_Figures/ET_along_resolution_bias.png'
# plt.savefig(outpath, dpi=600)
