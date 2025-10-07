import matplotlib; matplotlib.use('Qt5Agg')
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

## main ##
current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')
df = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/tree_ra_rs.csv')
df2 = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/grass_ra_rs.csv')

ra_tree = df['ra_tree']
ra_tree[ra_tree<10] = 10
ra_tree[ra_tree>500] = 500
rs_tree = df['rv_tree']*0.8
rs_tree[rs_tree<10] = 10
rs_tree[rs_tree>500] = 500

ra_grass = df2['ra_grass']
ra_grass[ra_grass<10] = 10
ra_grass[ra_grass>500] = 500
rs_grass = df2['rv_grass']*0.8
rs_grass[rs_grass<10] = 10
rs_grass[rs_grass>500] = 500


# Calculate means and standard deviations
ra_tree_mean, ra_tree_std = ra_tree.mean(), ra_tree.std()*0.25
ra_grass_mean, ra_grass_std = ra_grass.mean(), ra_grass.std()*0.25
rs_tree_mean, rs_tree_std = rs_tree.mean(), rs_tree.std()*0.25
rs_grass_mean, rs_grass_std = rs_grass.mean(), rs_grass.std()*0.25

# Data for plotting
labels = ['Aerodynamic resistance', 'Surface resistance']
ra_means = [ra_tree_mean, ra_grass_mean]
ra_stds = [ra_tree_std, ra_grass_std]
rs_means = [rs_tree_mean, rs_grass_mean]
rs_stds = [rs_tree_std, rs_grass_std]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# Create plot
fig, ax1 = plt.subplots(figsize=(10*0.5, 7*0.5))

# Bar plot for ra
color_ra = 'skyblue'
color_rs = 'lightcoral'
ax1.set_ylabel('Resistance values (s/m)',  fontsize=14)
rects1 = ax1.bar([-0.175, 0.825], [ra_tree_mean, rs_tree_mean], width, yerr=[ra_tree_std, rs_tree_std], color=color_ra, capsize=5, label='Tree')
rects2 = ax1.bar([0.175, 1.175], [ra_grass_mean, rs_grass_mean], width, yerr=[ra_grass_std, rs_grass_std], color=color_rs, capsize=5, label='Grass')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=12)
fig.tight_layout()  # otherwise the right y-label is slightly clipped


# Save figure
figure_path = os.path.join(current_dir, '4_Figures', 'Fig.S10.tree_vs_grass_rs_ra.png')
os.makedirs(os.path.dirname(figure_path), exist_ok=True) # Ensure directory exists
plt.savefig(figure_path, dpi=600)

plt.show()



