# 生成合成数据验证XGBoost端元解混能力
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as st

def xgb_unmix(ENDMEMBERS):
    # 2. 生成10,000个混合像元（随机土地覆盖比例）
    fractions = np.random.dirichlet(np.ones(4), 10000)  # 4类土地覆盖比例和=1
    mixed_LST = np.dot(fractions, list(ENDMEMBERS.values())) + np.random.normal(0, 1, 10000)
    # 3. 用XGBoost训练混合像元→端元LST的映射

    model = xgb.XGBRegressor(max_depth=2, min_child_weight=11, subsample=0.65, colsample_bytree=0.65, eta=0.15,
                             tree_method='hist', max_delta_step=6, eval_metric='auc')  # lower max_depth, larger

    model.fit(X=fractions, y=mixed_LST)
    # 4. 预测单类100%覆盖时的LST理论值
    pred_pure = model.predict([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    # 5. 计算反演误差：|预测端元值 - 真实端元值|
    return pred_pure

# 生成100组ENDMEMBERS
np.random.seed(42)
endmembers_list = []
for _ in range(1000):
    # 4类端元LST在合理范围内随机
    tree = np.random.uniform(25, 35)
    grass = np.random.uniform(27, 37)
    builtup = np.random.uniform(32, 42)
    crop = np.random.uniform(28, 38)
    endmembers_list.append({"tree": tree, "grass": grass, "builtup": builtup, "crop": crop})

# 记录真实值和预测值
true_vals = []
pred_vals = []

for em in endmembers_list:
    pred = xgb_unmix(em)
    true = list(em.values())
    pred_vals.append(pred)
    true_vals.append(true)

true_vals = np.array(true_vals)
pred_vals = np.array(pred_vals)

# 评估整体
r2 = r2_score(true_vals, pred_vals, multioutput='uniform_average')
rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
bias = np.mean(pred_vals - true_vals)

print(f'Overall R2: {r2:.3f}')
print(f'Overall RMSE: {rmse:.3f}')
print(f'Overall Bias: {bias:.3f}')

# 分别计算每个endmember的精度，并画scatter plot到一个大图
labels = ['tree', 'grass', 'builtup', 'crop']
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
axes = axes.flatten()

for i, label in enumerate(labels):
    r2_i = st.linregress(true_vals[:, i], pred_vals[:, i]).rvalue**2
    rmse_i = np.sqrt(mean_squared_error(true_vals[:, i], pred_vals[:, i]))
    bias_i = np.mean(pred_vals[:, i] - true_vals[:, i])

    ax = axes[i]
    ax.scatter(true_vals[:, i], pred_vals[:, i], alpha=0.6, label=None)

    # 计算线性回归拟合线
    slope, intercept, r_value, p_value, std_err = st.linregress(true_vals[:, i], pred_vals[:, i])
    x_fit = np.linspace(true_vals[:, i].min(), true_vals[:, i].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k-')  # 不在legend中显示

    # 画1:1线
    ax.plot(x_fit, x_fit, 'r--', linewidth=1)

    # 只在legend中显示统计信息
    ax.set_xlabel(f'True LST of {label}')
    ax.set_ylabel(f'Predicted LST of {label}')
    ax.set_title(f'{label}')
    ax.legend([f'y={slope:.2f}x+{intercept:.2f}\nR2={r2_i:.2f} \nRMSE={rmse_i:.2f}\nBias={bias_i:.2f}'])

plt.tight_layout()
plt.savefig('xgb_endmember_scatter.png', dpi=600)
# plt.close()