import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns;sns.set()

# read the data
stack_data = pd.read_csv("HyTL_data/stack_data.csv")
nut_data = pd.read_csv("HyTL_data/nut_data.csv")
cleanup_data = pd.read_csv("HyTL_data/cleanup_data.csv")
peg_data = pd.read_csv("HyTL_data/peg_data.csv")

TSBOARD_SMOOTHING = 0.96

stack_data   = stack_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
nut_data     = nut_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
cleanup_data = cleanup_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
peg_data = peg_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

stack_data   = np.array(stack_data)
nut_data     = np.array(nut_data)
cleanup_data = np.array(cleanup_data)
peg_data = np.array(peg_data)

stack_data0 = np.concatenate((stack_data[:,0]-stack_data[:,1],stack_data[:,0],stack_data[:,0] + stack_data[:,1]))
stack_data1 = np.concatenate((stack_data[:,2]-stack_data[:,3],stack_data[:,2],stack_data[:,2] + stack_data[:,3]))
stack_data2 = np.concatenate((stack_data[:,4]-stack_data[:,5],stack_data[:,4],stack_data[:,4] + stack_data[:,5]))
stack_data3 = np.concatenate((stack_data[:,6]-stack_data[:,7],stack_data[:,6],stack_data[:,6] + stack_data[:,7]))
stack_data4 = np.concatenate((stack_data[:,8]-stack_data[:,9],stack_data[:,8],stack_data[:,8] + stack_data[:,9]))

nut_data0 = np.concatenate((nut_data[:,0]-nut_data[:,1],nut_data[:,0],nut_data[:,0] + nut_data[:,1]))
nut_data1 = np.concatenate((nut_data[:,2]-nut_data[:,3],nut_data[:,2],nut_data[:,2] + nut_data[:,3]))
nut_data2 = np.concatenate((nut_data[:,4]-nut_data[:,5],nut_data[:,4],nut_data[:,4] + nut_data[:,5]))
nut_data3 = np.concatenate((nut_data[:,6]-nut_data[:,7],nut_data[:,6],nut_data[:,6] + nut_data[:,7]))
nut_data4 = np.concatenate((nut_data[:,8]-nut_data[:,9],nut_data[:,8],nut_data[:,8] + nut_data[:,9]))

cleanup_data0 = np.concatenate((cleanup_data[:,0]-cleanup_data[:,1],cleanup_data[:,0],cleanup_data[:,0] + cleanup_data[:,1]))
cleanup_data1 = np.concatenate((cleanup_data[:,2]-cleanup_data[:,3],cleanup_data[:,2],cleanup_data[:,2] + cleanup_data[:,3]))
cleanup_data2 = np.concatenate((cleanup_data[:,4]-cleanup_data[:,5],cleanup_data[:,4],cleanup_data[:,4] + cleanup_data[:,5]))
cleanup_data3 = np.concatenate((cleanup_data[:,6]-cleanup_data[:,7],cleanup_data[:,6],cleanup_data[:,6] + cleanup_data[:,7]))
cleanup_data4 = np.concatenate((cleanup_data[:,8]-cleanup_data[:,9],cleanup_data[:,8],cleanup_data[:,8] + cleanup_data[:,9]))

peg_data0 = np.concatenate((peg_data[:,0]-peg_data[:,1],peg_data[:,0],peg_data[:,0] + peg_data[:,1]))
peg_data1 = np.concatenate((peg_data[:,2]-peg_data[:,3],peg_data[:,2],peg_data[:,2] + peg_data[:,3]))
peg_data2 = np.concatenate((peg_data[:,4]-peg_data[:,5],peg_data[:,4],peg_data[:,4] + peg_data[:,5]))
peg_data3 = np.concatenate((peg_data[:,6]-peg_data[:,7],peg_data[:,6],peg_data[:,6] + peg_data[:,7]))
peg_data4 = np.concatenate((peg_data[:,8]-peg_data[:,9],peg_data[:,8],peg_data[:,8] + peg_data[:,9]))

stack_data_total = []
stack_data_total.append(stack_data0)
stack_data_total.append(stack_data1)
stack_data_total.append(stack_data2)
stack_data_total.append(stack_data3)
stack_data_total.append(stack_data4)

nut_data_total = []
nut_data_total.append(nut_data0)
nut_data_total.append(nut_data1)
nut_data_total.append(nut_data2)
nut_data_total.append(nut_data3)
nut_data_total.append(nut_data4)

cleanup_data_total = []
cleanup_data_total.append(cleanup_data0)
cleanup_data_total.append(cleanup_data1)
cleanup_data_total.append(cleanup_data2)
cleanup_data_total.append(cleanup_data3)
cleanup_data_total.append(cleanup_data4)

peg_data_total = []
peg_data_total.append(peg_data0)
peg_data_total.append(peg_data1)
peg_data_total.append(peg_data2)
peg_data_total.append(peg_data3)
peg_data_total.append(peg_data4)


print(stack_data_total)
print(nut_data_total)
print(cleanup_data_total)
print(peg_data_total)

stack_epoch1 = range(len(stack_data[:,0]))
stack_epoch2 = range(len(stack_data[:,0]))
stack_epoch3 = range(len(stack_data[:,0]))
stack_epoch = np.concatenate((stack_epoch1,stack_epoch2,stack_epoch3))
nut_epoch = stack_epoch

cleanup_epoch1 = range(len(cleanup_data[:,0]))
cleanup_epoch2 = range(len(cleanup_data[:,0]))
cleanup_epoch3 = range(len(cleanup_data[:,0]))
cleanup_epoch = np.concatenate((cleanup_epoch1,cleanup_epoch2,cleanup_epoch3))
peg_epoch = cleanup_epoch


labels = [
    'SAC',
    # 'SAC$_\mathrm{GNN-LTL}$',
    'MAPLE',
    'MAPLE$_\mathrm{Way}$',
    'TRAPs$_\mathrm{TF-LTL}$',
    'HyTL',
    # 'HyTL$'
]

color1 = sns.color_palette('deep')
color2 = sns.color_palette('muted')
color3=[
        color2[3],
        color2[0],
        color2[4],
        color2[1],
       color1[2],
       ]

lgd = plt.figure(figsize=(22, 5))

stack = plt.subplot(141)
# stack = plt.subplot(231)
for i in range(len(stack_data_total)):
    print(i)
    stack = sns.lineplot(x=stack_epoch, y=stack_data_total[i],label=labels[i], color=color3[i])
    # 　sns.color_palette("muted", 5)
plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# stack.set_xlim(0,)
stack.set_ylim(-0.5, )
plt.xlabel("Million Steps", fontsize=13)
plt.ylabel("Reward", fontsize=14)
plt.title("Stack", fontsize=16)
print('finsh stack')
stack.legend_.remove()

nut = plt.subplot(142)
for i in range(len(nut_data_total)):
    sns.lineplot(x=nut_epoch, y=nut_data_total[i],color=color3[i])
plt.xticks([0,500,1000,1500,2000,2500,3000],[0,0.5,1.0,1.5,2.0,2.5,3.0])
# plt.xlabel("million steps", fontsize=14)
# nut.set_xlim(0,)
nut.set_ylim(-0.5,)
# plt.xlabel("Million Steps", fontsize=13)
plt.ylabel("Reward", fontsize=14)
plt.title("Nut Assembly", fontsize=16)
print('finsh nut')

cleanup = plt.subplot(143)
for i in range(len(cleanup_data_total)):
    sns.lineplot(x=cleanup_epoch, y=cleanup_data_total[i],color=color3[i])
plt.xticks([0,500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
# plt.xlabel("million steps", fontsize=14)
# cleanup.set_xlim(0,)
cleanup.set_ylim(-0.5,)
# plt.xlabel("Million Steps", fontsize=13)
plt.ylabel("Reward", fontsize=14)
plt.title("Cleanup", fontsize=16)
print('finsh cleanup')

peg_into = plt.subplot(144)
for i in range(len(peg_data_total)):
    sns.lineplot(x=peg_epoch, y=peg_data_total[i],color=color3[i])
plt.xticks([0,500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
# plt.xlabel("million steps", fontsize=14)
# cleanup.set_xlim(0,)
cleanup.set_ylim(-0.5,)
# plt.xlabel("Million Steps", fontsize=13)
plt.ylabel("Reward", fontsize=14)
plt.title("Peg Insertion", fontsize=16)
print('finsh peg_into')


label = [
    'SAC',
    # 'SAC$_\mathrm{GNN-LTL}$',
    'MAPLE',
    'MAPLE$_\mathrm{Way}$',
    'TRAPs$_\mathrm{TF-LTL}$',
    'HyTL',
    # 'HyTL$'
]

print('lgd.axes is :', lgd.axes)
lines_labels = [ax.get_legend_handles_labels() for ax in lgd.axes]
print('lines_labels are:', lines_labels)
lines, label = [sum(lol, []) for lol in zip(*lines_labels)]

leg = lgd.legend([lines[0], lines[1], lines[2], lines[3]], labels,
                 loc='lower center',
                 ncol=5,
                 borderaxespad=-0.4,
                 frameon=False,
                 fontsize=14
                 )

for line in leg.get_lines():
    line.set_linewidth(4)

plt.tight_layout()
plt.savefig('plot_HyTL.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            )
plt.savefig('plot_HyTL.png',
            dpi=500,
            format='png',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')

print('save done')

plt.show()