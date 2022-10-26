import matplotlib.pyplot as plt
import numpy as np
import os
file_names = os.listdir('./new_results/')
# bo: 基础的做bo，上一次数据不会用到下次，上次先验也不用到下次
# linearize: 模型有线性化误差
data = []
for name in file_names:
    if name != 'tvbo.md' and name != 'tvbo_3.md' \
            and name != 'tvbo_transfer_1.md':
        with open('./new_results/' + name) as f:
            lines = f.readlines()
            lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
            lines = np.array(lines)
            data.append(lines)

        plt.plot(lines[1:], label=name.strip('.md'))

plt.legend()
plt.show()
