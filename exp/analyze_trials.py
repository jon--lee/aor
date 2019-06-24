import numpy as np
import matplotlib.pyplot as plt
import IPython
import os
import pickle

plt.style.use('ggplot')

def load_data(path, alg):
    filename = alg + '.p'
    filepath = os.path.join(path, filename)

    f = open(filepath, 'rb')
    results = pickle.load(f)
    f.close()

    costs = np.zeros((len(results), len(results[0]['rewards'])))
    for t, result in enumerate(results):
        costs[t, :] = result['rewards']

    return costs

alg = 'ig'
directories = ['data/cartpole_force_mag10.0', 'data/reg_cartpole_force_mag10.0']
paths = [os.path.join(direc, alg) for direc in directories]
datas = [load_data(path, alg) for path in paths]

rows = len(datas)
cols = datas[0].shape[0]


# plt.figure(figsize=(10,))
f, axarr = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(10, 3))
for i, data in enumerate(datas):
    for j, cost in enumerate(data):
        for _ in range(1):
            axarr[i, j].plot([50], cost[50])
        axarr[i, j].plot(np.arange(50, 100), cost[50:])
        axarr[i, j].set_ylim(0, 8)
        # axarr[i*cols, j].ylim(0, 15)
# f.subplots_adjust(hspace=0)
plt.tight_layout()
# f.text(0.5, 0.04, 'Iterations', ha='center')
# f.text(0.04, 0.5, 'Cost', va='center', rotation='vertical')
# plt.show()
plt.savefig('images/cartpole-ig.pdf')

