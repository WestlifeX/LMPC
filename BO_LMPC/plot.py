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
# 3系列是不加np.sign()的uncertainty，是ok的
# xxx_4: [np.clip(np.sign(xt[0]) * (np.exp(xt[0] ** 2 / 200) - 1), -0.2, 0.2),
#     np.clip(np.sign(xt[1]) * (-np.exp(xt[1] ** 2 / 200) + 1), -0.2, 0.2)]
# 如上所示的uncertainty，太大的就直接饱和了
for i in range(6):
    data = []
    for name in file_names:
        if name == 'robust_bo_{}.md'.format(i+1) or name == 'robust_tvbo_{}.md'.format(i+1):
            with open('./new_results/' + name) as f:
                lines = f.readlines()
                lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
                lines = np.array(lines)
                y = np.min(lines)
                data.append(lines)

            plt.plot(lines[0:50], label=name.strip('.md'))
            plt.plot(np.linspace(1, 50, 50), np.ones(50)*y)
    plt.legend()
    plt.show()
