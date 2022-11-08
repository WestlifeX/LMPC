import matplotlib.pyplot as plt
import numpy as np
import os
file_names = os.listdir('./new_results/')
# bo: 基础的做bo，上一次数据不会用到下次，上次先验也不用到下次
# linearize: 模型有线性化误差
# nonlinear: Ts=0.05, x0=[0.1, 0, 0.25, -0.01]，正确的非线性模型
# transfer_fg_1: Ts=0.05, x0=[0.1, 0, 0.25, -0.01]，线性模型，分段权重
# nonlinear_50: Ts=0.1, x0=[1, 0, 0.1, -0.01] # 50步
# transfer_fg_50: Ts=0.1, x0=[1, 0, 0.1, -0.01] # 50步


# robust_own: 基础的robust LMPC
# roubust_tvbo: fine-grained alpha tvbo + LMPC with 1+1
# robust_bo: simple bo with 1+1
data = []
for name in file_names:
    if name == 'robust_bo_3.md' or name == 'robust_tvbo_3.md' or name == 'robust_own_3.md':
        with open('./new_results/' + name) as f:
            lines = f.readlines()
            lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
            lines = np.array(lines)
            data.append(lines)

        plt.plot(lines[0:30], label=name.strip('.md'))

plt.legend()
plt.show()
