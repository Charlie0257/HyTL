import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns;sns.set()

#读取数据
lift_data = pd.read_csv("data/lift_data.csv")
door_data = pd.read_csv("data/door_data.csv")
pnp_data = pd.read_csv("data/pnp_data.csv")
stack_data = pd.read_csv("data/stack_data.csv")
nut_data = pd.read_csv("data/nut_data.csv")
cleanup_data = pd.read_csv("data/cleanup_data.csv")

TSBOARD_SMOOTHING = 0.96

lift_data    = lift_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
door_data    = door_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
pnp_data     = pnp_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
stack_data   = stack_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
nut_data     = nut_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
cleanup_data = cleanup_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

lift_data    = np.array(lift_data)
door_data    = np.array(door_data)
pnp_data     = np.array(pnp_data)
stack_data   = np.array(stack_data)
nut_data     = np.array(nut_data)
cleanup_data = np.array(cleanup_data)

lift_data0 = np.concatenate((lift_data[:,0]-lift_data[:,1],lift_data[:,0],lift_data[:,0] + lift_data[:,1]))
lift_data1 = np.concatenate((lift_data[:,2]-lift_data[:,3],lift_data[:,2],lift_data[:,2] + lift_data[:,3]))
lift_data2 = np.concatenate((lift_data[:,4]-lift_data[:,5],lift_data[:,4],lift_data[:,4] + lift_data[:,5]))
lift_data3 = np.concatenate((lift_data[:,6]-lift_data[:,7],lift_data[:,6],lift_data[:,6] + lift_data[:,7]))
lift_data4 = np.concatenate((lift_data[:,8]-lift_data[:,9],lift_data[:,8],lift_data[:,8] + lift_data[:,9]))

door_data0 = np.concatenate((door_data[:,0]-door_data[:,1],door_data[:,0],door_data[:,0] + door_data[:,1]))
door_data1 = np.concatenate((door_data[:,2]-door_data[:,3],door_data[:,2],door_data[:,2] + door_data[:,3]))
door_data2 = np.concatenate((door_data[:,4]-door_data[:,5],door_data[:,4],door_data[:,4] + door_data[:,5]))
door_data3 = np.concatenate((door_data[:,6]-door_data[:,7],door_data[:,6],door_data[:,6] + door_data[:,7]))
door_data4 = np.concatenate((door_data[:,8]-door_data[:,9],door_data[:,8],door_data[:,8] + door_data[:,9]))

pnp_data0 = np.concatenate((pnp_data[:,0]-pnp_data[:,1],pnp_data[:,0],pnp_data[:,0] + pnp_data[:,1]))
pnp_data1 = np.concatenate((pnp_data[:,2]-pnp_data[:,3],pnp_data[:,2],pnp_data[:,2] + pnp_data[:,3]))
pnp_data2 = np.concatenate((pnp_data[:,4]-pnp_data[:,5],pnp_data[:,4],pnp_data[:,4] + pnp_data[:,5]))
pnp_data3 = np.concatenate((pnp_data[:,6]-pnp_data[:,7],pnp_data[:,6],pnp_data[:,6] + pnp_data[:,7]))
pnp_data4 = np.concatenate((pnp_data[:,8]-pnp_data[:,9],pnp_data[:,8],pnp_data[:,8] + pnp_data[:,9]))

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

lift_data_total = []
lift_data_total.append(lift_data0)
lift_data_total.append(lift_data1)
lift_data_total.append(lift_data2)
lift_data_total.append(lift_data3)
lift_data_total.append(lift_data4)

door_data_total = []
door_data_total.append(door_data0)
door_data_total.append(door_data1)
door_data_total.append(door_data2)
door_data_total.append(door_data3)
door_data_total.append(door_data4)

pnp_data_total = []
pnp_data_total.append(pnp_data0)
pnp_data_total.append(pnp_data1)
pnp_data_total.append(pnp_data2)
pnp_data_total.append(pnp_data3)
pnp_data_total.append(pnp_data4)

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

print(lift_data_total)
print(door_data_total)
print(pnp_data_total)
print(stack_data_total)
print(nut_data_total)
print(cleanup_data_total)

lift_epoch1 = range(len(lift_data[:,0]))
lift_epoch2 = range(len(lift_data[:,0]))
lift_epoch3 = range(len(lift_data[:,0]))
lift_epoch = np.concatenate((lift_epoch1,lift_epoch2,lift_epoch3))
door_epoch = lift_epoch
pnp_epoch  = lift_epoch

stack_epoch1 = range(len(stack_data[:,0]))
stack_epoch2 = range(len(stack_data[:,0]))
stack_epoch3 = range(len(stack_data[:,0]))
stack_epoch = np.concatenate((stack_epoch1,stack_epoch2,stack_epoch3))
nut_epoch = stack_epoch

cleanup_epoch1 = range(len(cleanup_data[:,0]))
cleanup_epoch2 = range(len(cleanup_data[:,0]))
cleanup_epoch3 = range(len(cleanup_data[:,0]))
cleanup_epoch = np.concatenate((cleanup_epoch1,cleanup_epoch2,cleanup_epoch3))

labels = [
    'SAC',
    'SAC$_\mathrm{GNN-LTL}$',
    'MAPLE',
    'TRAPs$_\mathrm{GNN-LTL}$',
    'TRAPs$_\mathrm{TF-LTL}$',
    # 'TL2$_\mathrm{pre}$'
]

color1 = sns.color_palette('deep')
color2 = sns.color_palette('muted')
color3=[color2[0],color2[1],color2[4],color2[3],color1[2]]

lgd = plt.figure(figsize=(16, 9))

lift = plt.subplot(231)
# lift.set_ylim(-0.5,)
for i in range(len(lift_data_total)):
    print(i)
    lift=sns.lineplot(x=lift_epoch, y=lift_data_total[i],label=labels[i],color=color3[i])

plt.xticks([0,500,1000,1500,2000,2500],[0,0.5,1.0,1.5,2.0,2.5])
# lift.set_xlim(0,)
lift.set_ylim(-0.5,)
plt.xlabel("Million Steps", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.title("Lift", fontsize=16)
print('finsh lift')
lift.legend_.remove()

# plt.subplot(232)
# # plt.xlabel("million steps", fontsize=14)
# plt.ylabel("reward", fontsize=14)
# plt.title("Door", fontsize=16)
# print('finsh door')
# #
# plt.subplot(233)
# # plt.xlabel("million steps", fontsize=14)
# plt.ylabel("reward", fontsize=14)
# plt.title("Pick and Place", fontsize=16)
# print('finsh pnp')

door = plt.subplot(232)
# plt.set_ylim(-0.5,)
for i in range(len(door_data_total)):
    sns.lineplot(x=door_epoch, y=door_data_total[i],color=color3[i])
    # sns.color_palette("muted", 5)
plt.xticks([0,500,1000,1500,2000,2500],[0,0.5,1.0,1.5,2.0,2.5])
# plt.xlabel("million steps", fontsize=14)
# door.set_xlim(0,)
door.set_ylim(-0.5,)
plt.ylabel("Reward", fontsize=14)
plt.title("Door Opening", fontsize=16)
print('finsh door')

pnp = plt.subplot(233)
for i in range(len(pnp_data_total)):
    sns.lineplot(x=pnp_epoch, y=pnp_data_total[i],color=color3[i])
plt.xticks([0,500,1000,1500,2000,2500],[0,0.5,1.0,1.5,2.0,2.5])
# plt.xlabel("million steps", fontsize=14)
# pnp.set_xlim(0,)
pnp.set_ylim(-0.5,)
plt.ylabel("Reward", fontsize=14)
plt.title("Pick and Place", fontsize=16)
print('finsh pnp')
#
# plt.subplot(234)
# # plt.xlabel("million steps", fontsize=14)
# plt.ylabel("reward", fontsize=14)
# plt.title("Stack", fontsize=16)
# print('finsh stack')
#
# plt.subplot(235)
# # plt.xlabel("million steps", fontsize=14)
# plt.ylabel("reward", fontsize=14)
# plt.title("Nut", fontsize=16)
# print('finsh nut')
# # #
# plt.subplot(236)
# # plt.xlabel("million steps", fontsize=14)
# plt.ylabel("reward", fontsize=14)
# plt.title("Cleanup", fontsize=16)
# print('finsh cleanup')

stack = plt.subplot(234)
for i in range(len(stack_data_total)):
    sns.lineplot(x=stack_epoch, y=stack_data_total[i],color=color3[i])
    sns.color_palette("muted", 5)
plt.xticks([0,500,1000,1500,2000,2500,3000],[0,0.5,1.0,1.5,2.0,2.5,3.0])
# stack.set_xlim(0,)
stack.set_ylim(-0.5,)
# plt.xlabel("million steps", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.title("Stack", fontsize=16)
print('finsh stack')

nut = plt.subplot(235)
for i in range(len(nut_data_total)):
    sns.lineplot(x=nut_epoch, y=nut_data_total[i],color=color3[i])
plt.xticks([0,500,1000,1500,2000,2500,3000],[0,0.5,1.0,1.5,2.0,2.5,3.0])
# plt.xlabel("million steps", fontsize=14)
# nut.set_xlim(0,)
nut.set_ylim(-0.5,)
plt.ylabel("Reward", fontsize=14)
plt.title("Nut Assembly", fontsize=16)
print('finsh nut')
#
cleanup = plt.subplot(236)
for i in range(len(cleanup_data_total)):
    sns.lineplot(x=cleanup_epoch, y=cleanup_data_total[i],color=color3[i])
plt.xticks([0,500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
# plt.xlabel("million steps", fontsize=14)
# cleanup.set_xlim(0,)
cleanup.set_ylim(-0.5,)
plt.ylabel("Reward", fontsize=14)
plt.title("Cleanup", fontsize=16)
print('finsh cleanup')

label = [
    'SAC',
    'SAC+GNN',
    'MAPLE',
    'TRAPs$_\mathrm{GNN}$',
    'TRAPs$_\mathrm{TF-LTL}$',
    # 'TL2$_\mathrm{pre}$'
]

# handles, labels = lift.get_legend_handles_labels()
# lgd.legend(handles, labels, loc='upper right', ncol=3, bbox_to_anchor=(.75, 0.98))
lines_labels = [ax.get_legend_handles_labels() for ax in lgd.axes]
lines, label = [sum(lol, []) for lol in zip(*lines_labels)]

leg=lgd.legend([lines[0],lines[1],lines[2],lines[3],lines[4]], labels,
           loc='lower center',
           ncol=5,
           borderaxespad=-0.4,
           frameon=False,
           fontsize=14
           )
for line in leg.get_lines():
    line.set_linewidth(4)
# lgd.legend([lift0, lift1, lift2, lift3, lift4], label, loc='lower center', ncol=5)

plt.tight_layout()
plt.savefig('date.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            )
plt.savefig('data.png',
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

