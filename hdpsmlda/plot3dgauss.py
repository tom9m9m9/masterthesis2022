import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
name='aist4'
mu=np.loadtxt('C:/Users/tom/src/hdpsmlda/output/mu_{}.txt'.format(name))
mu_true_kd=np.zeros([mu.shape[0],3])
print(mu)
fig = plt.figure()
ax1 = Axes3D(fig)
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['mathtext.default'] = 'it'

# 軸のラベルを設定する。
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$x_3$')
plt.rcParams['font.size'] = 14
#plt.rcParams['font.family'] = 'Times New Roman'

from numpy.random import *
cl=['black','lightsalmon','darkcyan','red','rosybrown',
'saddlebrown','gold','darkkhaki','olivedrab','lawngreen',
'turquoise','deepskyblue','navy','blue','blueviolet','magenta']

for i in range(len(mu)):
    # データを用意する
    sigmas=[[1,0,0],[0,1,0],[0,0,1]]
    mus=mu[i,:]
    values = multivariate_normal(mus, sigmas, 100)



    # データプロットする。
    ax1.scatter3D(values[:,0], values[:,1], values[:,2], label='Topic {}'.format(i+1),c=cl[i])

plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=18)
# グラフを表示する。
plt.show()

