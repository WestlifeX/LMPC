import matplotlib.pyplot as plt
import numpy as np
import os
file_names = os.listdir('./new_results/')
# file_names = os.listdir('./new_result_2/')
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
N = 6
N_alg = 4
end = 30
all = np.zeros((N_alg, end))
all_best = np.zeros((N_alg, end))
bo_data = np.zeros((N, end))
tlbo_data = np.zeros((N, end))
direct_data = np.zeros((N, end))
unlim_data = np.zeros((N, end))
idxs = [1, 2, 3, 4, 5, 6]

for i in range(N):
    idx = idxs[i]
    data = []
    for name in file_names:
        if name == 'robust_bo_{}.md'.format(idx) or name == 'robust_tvbo_{}.md'.format(idx) or \
                name == 'robust_tvbo_direct_{}.md'.format(idx) \
                or name == 'robust_tvbo_unlim_{}.md'.format(idx):
            with open('./new_results/' + name) as f:
                lines = f.readlines()
                lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
                lines = np.array(lines)
                current_best = np.zeros_like(lines)
                vmin = np.inf
                for j in range(lines.shape[0]):
                    if lines[j] < vmin:
                        current_best[j] = lines[j]
                        vmin = lines[j]
                    else:
                        current_best[j] = vmin
                y = np.min(lines)

                if name == 'robust_bo_{}.md'.format(idx):
                    all[0, :end] += lines[:end] / N
                    all_best[0, :end] += current_best[:end] / N
                    bo_data[i - 1, :end] = current_best[:end]
                elif name == 'robust_tvbo_{}.md'.format(idx):
                    all[1, :end] += lines[:end] / N
                    all_best[1, :] += current_best[:end] / N
                    tlbo_data[i - 1, :] = current_best[:end]
                elif name == 'robust_tvbo_direct_{}.md'.format(idx):
                    all[2, :end] += lines[:end] / N
                    all_best[2, :] += current_best[:end] / N
                    direct_data[i - 1, :] = current_best[:end]
                else:
                    all[3, :end] += lines[:end] / N
                    all_best[3, :] += current_best[:end] / N
                    unlim_data[i - 1, :] = current_best[:end]

    #         plt.plot(lines[0:50], label=name.strip('.md'))
    #         plt.plot(np.linspace(1, 50, 50), np.ones(50)*y)
    # plt.legend()
    # plt.show()
bo_std = np.std(bo_data, axis=0)
tlbo_std = np.std(tlbo_data, axis=0)
direct_std = np.std(direct_data, axis=0)
unlim_std = np.std(unlim_data, axis=0)

tlbo_max = np.max(tlbo_data, axis=0).reshape(1, -1)
tlbo_min = np.min(tlbo_data, axis=0).reshape(1, -1)
tlbo_error = np.vstack((all_best[1]-tlbo_min, tlbo_max-all_best[1]))
# plt.plot(all[0])
# plt.plot(all[1])
# plt.plot(all[2])
# plt.show()
colors = ['sandybrown', 'purple', 'green', 'steelblue']
alpha = 0.1
x = np.linspace(1, 30, 30)
plt.plot(x, all_best[0], label='Generic BO', lw=2, color=colors[0])
# plt.step(x, all_best[0], label='generic bo', lw=2, color='purple')
# plt.errorbar(np.arange(all_best.shape[1]), all_best[0], bo_std, capsize=3)
plt.fill_between(range(1, all_best.shape[1]+1), all_best[0]-bo_std, all_best[0]+bo_std, alpha=alpha,
                 facecolor=colors[0], edgecolor='none')
plt.plot(x, all_best[1], label='Efficient BO', lw=2, color=colors[1])
# plt.step(x, all_best[1], label='transfer learning bo', lw=2, color='slateblue')
# plt.errorbar(np.arange(1, all_best.shape[1]+1), all_best[1], tlbo_error, fmt='none', elinewidth=1,
#              capsize=3, ecolor=colors[1])
plt.fill_between(range(1, all_best.shape[1]+1), all_best[1]-tlbo_std, all_best[1]+tlbo_std, alpha=alpha,
                 facecolor=colors[1], edgecolor='none')
# plt.plot(all_best[2], label='lmpc')

plt.plot(x, all_best[2], label='Unnormalized Efficient BO', lw=2, color=colors[2])
plt.fill_between(range(1, all_best.shape[1]+1), all_best[2]-direct_std, all_best[2]+direct_std, alpha=alpha,
                 facecolor=colors[2], edgecolor='none')

plt.plot(x, all_best[3], label='Unbounded Efficient BO', lw=2, color=colors[3])
plt.fill_between(range(1, all_best.shape[1]+1), all_best[3]-unlim_std, all_best[3]+unlim_std, alpha=alpha,
                 facecolor=colors[3], edgecolor='none')
# plt.plot(all_best[3], label='tvbo all')
# plt.errorbar(np.arange(all_best.shape[1]), all_best[1], bo_std, capsize=3)
# plt.fill_between(range(all_best.shape[1]), all_best[3]-tlbo_all_std, all_best[3]+tlbo_all_std, alpha=0.3)
font = {'family': 'Times New Roman', 'size': 16}
plt.xlabel('Iterations', font)
plt.ylabel('Cost', font)

# 不显示右侧和上面的边框线
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.legend(prop=font)
plt.grid()
plt.savefig('./figs/cost.png', dpi=600)
plt.show()

n = 2
for i in range(30):
    print(str(i+1) + ' & ' + str(round(all_best[1, i], n)) + '(' + str(round(all_best[1, i]-tlbo_std[i], n))
          + ', ' + str(round(all_best[1, i]+tlbo_std[i], n)) + ')'
          + ' & ' + str(round(all_best[0, i], n)) + '(' + str(round(all_best[0, i]-bo_std[i], n))
          + ', ' + str(round(all_best[0, i]+bo_std[i], n)) + ')' +
          ' & ' + str(round(all_best[2, i], n)) + '(' + str(round(all_best[2, i]-direct_std[i], n))
          + ', ' + str(round(all_best[2, i]+direct_std[i], n)) + ')' +
          ' & ' + str(round(all_best[3, i], n)) + '(' + str(round(all_best[3, i]-unlim_std[i], n))
          + ', ' + str(round(all_best[3, i]+unlim_std[i], n)) + ')' + ' \\\\')
