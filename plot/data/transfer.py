import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import seaborn as sns;sns.set()

# def smooth(data, sm=1):
#     smooth_data = []
#     if sm > 1:
#         for d in data:
#             z = np.ones(len(d))
#             y = np.ones(sm)*1.0
#             d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
#             smooth_data.append(d)
#     return smooth_data
# def smooth(data, sm=1):
#     if sm > 1:
#         smooth_data = []
#         for d in data:
#             y = np.ones(sm)*1.0/sm
#             d = np.convolve(y, d, "same")
#
#             smooth_data.append(d)
#
#     return smooth_data
#
# def gaussian_filter1d(size,sigma):
#     filter_range = np.linspace(-int(size/2),int(size/2),size)
#     gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
#     return gaussian_filter

#读取数据
trans_data = pd.read_csv("data/trans.csv")
# door_data = pd.read_csv("data/door_data.csv")
# pnp_data = pd.read_csv("data/pnp_data.csv")
# stack_data = pd.read_csv("data/stack_data.csv")
# nut_data = pd.read_csv("data/nut_data.csv")
# cleanup_data = pd.read_csv("data/cleanup_data.csv")

TSBOARD_SMOOTHING = 0.99
#
trans_data    = trans_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

trans_data    = np.array(trans_data)
# print(trans_data)
# trans_data = trans_data.T
# print(trans_data)
# trans_data[0, :]    = gaussian_filter1d(trans_data[0, :], 25)
# trans_data[1, :]    = gaussian_filter1d(trans_data[1, :], 25)
# trans_data[2, :]    = gaussian_filter1d(trans_data[2, :], 25)
# trans_data[3, :]    = gaussian_filter1d(trans_data[3, :], 25)
#
# trans_data = trans_data.T
# print(trans_data)

lift_data0 = np.concatenate((trans_data[:,0]-trans_data[:,1],trans_data[:,0],trans_data[:,0] + trans_data[:,1]))
lift_data1 = np.concatenate((trans_data[:,2]-trans_data[:,3],trans_data[:,2],trans_data[:,2] + trans_data[:,3]))
# lift_data2 = np.concatenate((lift_data[:,4]-lift_data[:,5],lift_data[:,4],lift_data[:,4] + lift_data[:,5]))
# lift_data3 = np.concatenate((lift_data[:,6]-lift_data[:,7],lift_data[:,6],lift_data[:,6] + lift_data[:,7]))
# lift_data4 = np.concatenate((lift_data[:,8]-lift_data[:,9],lift_data[:,8],lift_data[:,8] + lift_data[:,9]))


trans_data_total = []
trans_data_total.append(lift_data0)
trans_data_total.append(lift_data1)

# trans_data_total    = smooth(trans_data_total, 300)

print(trans_data_total)


trans_epoch1 = range(len(trans_data[:, 0]))
trans_epoch2 = range(len(trans_data[:, 0]))
trans_epoch3 = range(len(trans_data[:, 0]))
trans_epoch = np.concatenate((trans_epoch1, trans_epoch2, trans_epoch3))

labels = [
    'From Scratch',
    'Transfer Sketch',
]

color1 = sns.color_palette('deep')
color2 = sns.color_palette('muted')
color3=[color2[0],color2[1],color2[4],color2[3],color1[2]]

lgd = plt.figure(figsize=(8, 6))

for i in range(len(trans_data_total)):
    print(i)
    lift=sns.lineplot(x=trans_epoch, y=trans_data_total[i],label=labels[i],color=color3[i],ci=50)

plt.xticks([0,500,1000,1500,2000,2500],[0,0.5,1.0,1.5,2.0,2.5], fontsize=16)
plt.yticks([0,0.25,0.5,0.75,1.00],[0,0.25,0.5,0.75,1.00], fontsize=16)
# lift.set_xlim(0,)
lift.set_ylim(-0.05,)
# plt.yticks(fontsize=16)
plt.xlabel("Million Steps", fontsize=16)
plt.ylabel("Success Rate", fontsize=16)
plt.legend(loc='lower right', fontsize=14, borderaxespad=1.5)
plt.title("Pick and Place Bread", fontsize=18)
print('finsh lift')
# lift.legend_.remove()


# lines_labels = [ax.get_legend_handles_labels() for ax in lgd.axes]
# lines, label = [sum(lol, []) for lol in zip(*lines_labels)]
#
# leg=lgd.legend([lines[0],lines[1],lines[2],lines[3],lines[4]], labels,
#            loc='lower center',
#            ncol=5,
#            borderaxespad=-0.4,
#            frameon=False,
#            fontsize=14
#            )
# for line in leg.get_lines():
#     line.set_linewidth(4)
# lgd.legend([lift0, lift1, lift2, lift3, lift4], label, loc='lower center', ncol=5)

plt.tight_layout()

plt.savefig('trans1.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            )
plt.savefig('trans1.png',
            dpi=500,
            format='png',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')

print('save done')

plt.show()



# #图例
# label = [
#     'DFA',
#     'RM',
#     'GNN$_\mathrm{pre}$',
#     'TL1$_\mathrm{pre}$',
#     # 'TL2$_\mathrm{pre}$'
# ]
#
# #设置颜色
# color = [
#     [0.7, 0.4, 0.2],
#     [0, 0.4, 0.9],
#     [0.8, 0.2, 0.9],
#     [1, 0.6, 0],
#     # [0.2, 0.7, 0]
# ]
#
# for i in range(len(data_total)):
#     sns.tsplot(time=xdata,
#                data=data_total[i],
#                color=color[i],
#                linestyle=None,
#                condition=label[i])
# plt.ylabel("Performance", fontsize=14)
# ax = plt.gca()
# plt.grid(alpha=0.3)
# ax.yaxis.set_ticks_position('both')
# plt.xlabel("Steps", fontsize=13)
# plt.title("Safety Gym", fontsize=16)
# plt.show()

