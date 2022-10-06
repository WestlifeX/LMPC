import matplotlib.pyplot as plt
import numpy as np
import os
file_names = os.listdir('./results/')

data = []
for name in file_names:
    with open('./results/' + name) as f:
        lines = f.readlines()
        lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
        lines = np.array(lines)
        data.append(lines)

    plt.plot(lines[1:], label=name.strip('.md'))

plt.legend()
plt.show()
