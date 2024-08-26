#
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from matplotlib import style
import numpy as np
import seaborn as sns
style.use('ggplot')  # 加载'ggplot'风格

# 加载中文字体
# font_path = "/System/Library/Fonts/STHeiti Light.ttc" # 本地字体链接
# prop = mfm.FontProperties(fname=font_path)

A = [[1, 1, 0.81, 0.71, 0.75, 0.73],
     [1, 1, 0.87, 0.79, 0.85, 0.75],
     [1, 1, 1,    0.92, 1,    0.83]]

x_labels = ['Lift', 'Door Opening', 'Pick and Place', 'Stack', 'Nut Assembly', 'Cleanup']

x = np.arange(6)
# 生成多柱图
# lgd = plt.figure(figsize=(10,3))

fig, ax = plt.subplots(figsize=(8.5,3.2))
color1 = sns.color_palette('deep')
color2 = sns.color_palette('muted')
color3=[color2[0],color2[1],color2[4],color2[3],color1[2]]
sns.set_style('white')
ax.bar(x + 0.00, A[0], color=color3[2], width=0.2, label='MAPLE')
ax.bar(x + 0.2, A[1], color=color3[3], width=0.2, label='TRAPs$_\mathrm{GNN-LTL}$')
ax.bar(x + 0.4, A[2], color=color3[4], width=0.2, label='TRAPs$_\mathrm{TF-LTL}$')

plt.xticks(x + 0.20, x_labels, fontsize=12, color='black')
plt.xlabel('Environments',fontsize=14,color='black')
plt.ylim( 0.0 ,1.18)
plt.yticks(fontsize=12,color='black')
plt.ylabel('Compositionality Score', fontsize=14,color='black')
# lgd.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# 生成图片
plt.legend(fontsize=12,
           loc='upper center',
           borderaxespad=0.2,
           frameon=False,
           # loc="upper right",
           ncol=3)

plt.tight_layout()
plt.savefig("score.png", dpi=700)
plt.savefig('score.pdf',
            format='pdf',
            bbox_extra_artists=(fig,),
            bbox_inches='tight',
            )
plt.show()

import pandas as pd
# import csv
#
# import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # comp = pd.read_csv("./comp_data.csv")
# tips = pd.read_csv('./comp_data.csv', encoding='UTF-8')
# # tips = tips.head()
# print(tips)
# env=['Lift','Door', 'Pick and Place','Stack','Nut','Cleanup']
# score=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# # lgd = plt.figure()
#
# fig, ay = plt.subplots(figsize=(10, 4))
#
# labels=['MAPLE', 'TRAPs$_\mathrm{GNN}$', 'TRAPs$_\mathrm{TF-LTL}$']
#
# ax=sns.barplot(x="Environments", y="Compositionality Score", hue="Methods", data=tips)
# lab=ax.label()
# print(lab)
#
# ax.set_xticklabels(env, fontsize=14)
# ax.set_yticklabels(score, fontsize=14)
# ax.set_xlabel("Environments", fontsize=16)
# ax.set_ylabel("Compositionality Score", fontsize=16)
#
# ay.legend_.remove()
# # h = ax.get_legend().legendHandles
# h, _ = ax.get_legend_handles_labels()
#
# # print(h)
# ay.legend(h, ['MAPLE', 'TRAPs$_\mathrm{GNN}$', 'TRAPs$_\mathrm{TF-LTL}$'])
# # ax.legend(h, ['1', '2', '3'])
# plt.gca().legend().set_title('')
# # plt.xlabel(x="env", fontsize=14)
# # plt.ylabel(y="data", fontsize=14)
#
# plt.show()
# #