import matplotlib.pyplot as plt
import numpy as np
import os
file_names = os.listdir('./results/')
# bo: 基础的做bo，上一次数据不会用到下次，上次先验也不用到下次
# linear: 系统没有误差，实际就是线性模型
# nonlinear: 实际系统是非线性的，但做了线性化
# tvbo: 最简单的tvbo，n_initial_points & n_iters 均为5，边界为200（好像）
# tvbo_2: 扩大数据量，n_initial_points & n_iters 均为10，边界为1000
# tvbo_3: 减小数据量，n_initial_points & n_iters 均为1，边界为1000
# tvbo_transfer_1: 利用之前数据时加上了权重，给过去的所有数据都加一样且固定的权重，n_initial_points=1，n_iters=3
# tvbo_transfer_2: 在transfer_1的基础上取消后验变先验的设定（还没跑）
# tvbo_transfer_fg_2: 每段数据的权重大小是不一样的
data = []
for name in file_names:
    if name != 'linear.md' and name != 'tvbo.md':
        with open('./results/' + name) as f:
            lines = f.readlines()
            lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
            lines = np.array(lines)
            data.append(lines)

        plt.plot(lines[1:], label=name.strip('.md'))

plt.legend()
plt.show()
