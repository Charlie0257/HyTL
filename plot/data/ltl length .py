import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns;sns.set()

#读取数据

cleanup_data = pd.read_csv("data/ltl length1.csv")

TSBOARD_SMOOTHING = 0.99

cleanup_data = cleanup_data.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

cleanup_data = np.array(cleanup_data)


cleanup_data0 = np.concatenate((cleanup_data[:,0]-cleanup_data[:,1],cleanup_data[:,0],cleanup_data[:,0] + cleanup_data[:,1]))
cleanup_data1 = np.concatenate((cleanup_data[:,2]-cleanup_data[:,3],cleanup_data[:,2],cleanup_data[:,2] + cleanup_data[:,3]))
cleanup_data2 = np.concatenate((cleanup_data[:,4]-cleanup_data[:,5],cleanup_data[:,4],cleanup_data[:,4] + cleanup_data[:,5]))
# cleanup_data3 = np.concatenate((cleanup_data[:,6]-cleanup_data[:,7],cleanup_data[:,6],cleanup_data[:,6] + cleanup_data[:,7]))
# cleanup_data4 = np.concatenate((cleanup_data[:,8]-cleanup_data[:,9],cleanup_data[:,8],cleanup_data[:,8] + cleanup_data[:,9]))
# cleanup_data0 = np.concatenate((cleanup_data[:,0],cleanup_data[:,0],cleanup_data[:,0]))
# cleanup_data1 = np.concatenate((cleanup_data[:,2],cleanup_data[:,2],cleanup_data[:,2]))
# cleanup_data2 = np.concatenate((cleanup_data[:,4],cleanup_data[:,4],cleanup_data[:,4]))
# cleanup_data3 = np.concatenate((cleanup_data[:,6],cleanup_data[:,6],cleanup_data[:,6]))
# cleanup_data4 = np.concatenate((cleanup_data[:,8],cleanup_data[:,8],cleanup_data[:,8]))

cleanup_data_total = []
cleanup_data_total.append(cleanup_data0)
cleanup_data_total.append(cleanup_data1)
cleanup_data_total.append(cleanup_data2)
# cleanup_data_total.append(cleanup_data3)
# cleanup_data_total.append(cleanup_data4)


print(cleanup_data_total)


cleanup_epoch1 = range(len(cleanup_data[:,0]))
cleanup_epoch2 = range(len(cleanup_data[:,0]))
cleanup_epoch3 = range(len(cleanup_data[:,0]))
cleanup_epoch = np.concatenate((cleanup_epoch1,cleanup_epoch2,cleanup_epoch3))

labels = [
    '2AP+3OP',
    '3AP+5OP',
    '5AP+11OP',
    # '$\lambda$=0.75',
    # '$\lambda$=1.0',
    # 'TL2$_\mathrm{pre}$'
]

color1 = sns.color_palette('deep')
color2 = sns.color_palette('muted')
color3=[color1[2],color2[1],color2[4],color2[0],color2[3]]

lgd = plt.figure(figsize=(8, 6))

for i in range(len(cleanup_data_total)):
    cleanup=sns.lineplot(x=cleanup_epoch, y=cleanup_data_total[i],label=labels[i],color=color3[i],ci=45)
plt.xticks([0,500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])

plt.xlabel("million steps", fontsize=24)
# cleanup.set_xlim(0,)
cleanup.set_ylim(-0.05,)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("Million Steps", fontsize=24)
plt.ylabel("Success Rate", fontsize=24)
plt.legend(loc='lower right', fontsize=20, borderaxespad=1.5)
# plt.title("Cleanup", fontsize=16)
print('finsh cleanup')

plt.tight_layout()
plt.savefig('ltl_length2.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            )
plt.savefig('ltl_length2.png',
            dpi=500,
            format='png',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')

print('save done')

plt.show()
