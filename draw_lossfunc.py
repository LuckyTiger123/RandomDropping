import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams['font.sans-serif'] = ['Times New Roman']


def get_xdata(filename, dropping_method):
    df = pd.read_excel(filename.format(dropping_method))
    dd = df.to_numpy()
    dd = dd.transpose()
    xx = dd[1][10:500]  # epoch
    return xx


def get_ydata(filename, dropping_method):
    print('open <' + filename.format(dropping_method) + '> !')

    df = pd.read_excel(filename.format(dropping_method))
    dd = df.to_numpy()
    dd = dd.transpose()
    yy = dd[2][10:500]  # loss
    return yy


dataset = 'Cora'
backbone = 'GCN'
dropping_rate = '0.7'

filepath = './result/' + 'statistic/' + dataset + '/'
filename = filepath + backbone + '_{}_' + str(dropping_rate) + '_lossfunc_without_normalization.xlsx'

dropping_method_list = ['Clean', 'DropEdge', 'Dropout', 'DropNode', 'DropMessage']
# dropping_method_list = ['Clean', 'DropEdge', 'Dropout', 'DropMessage']

xx = get_xdata(filename, dropping_method_list[0])
yy_list = []
yy_list.append(get_ydata(filename, dropping_method_list[0]))
yy_list.append(get_ydata(filename, dropping_method_list[1]))
yy_list.append(get_ydata(filename, dropping_method_list[2]))
yy_list.append(get_ydata(filename, dropping_method_list[3]))
yy_list.append(get_ydata(filename, dropping_method_list[4]))

fig, ax = plt.subplots()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.plot(xx, yy_list[0], label=backbone + '-' + dropping_method_list[0])
ax.plot(xx, yy_list[1], label=backbone + '-' + dropping_method_list[1])
ax.plot(xx, yy_list[2], label=backbone + '-' + dropping_method_list[2])
ax.plot(xx, yy_list[3], label=backbone + '-' + dropping_method_list[3])
ax.plot(xx, yy_list[4], label=backbone + '-' + dropping_method_list[4])
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Loss', fontsize=16)
# ax.set_title('Test loss on {} dataset'.format(dataset))
ax.legend(fontsize=16)

plt.show()
# fig.show()
fig.savefig('./result/vector_graph/scatter.pdf', dpi=600, format='pdf')
# savepath = './fig_loss/' + dataset + '/'
# if not os.path.exists(savepath):
#     os.mkdir(savepath)
#
# fig.savefig(os.path.join(savepath, '{}_{}_lossfunc.svg'.format(backbone, dropping_rate)))
# print('save <' + os.path.join(savepath, '{}_{}_lossfunc.svg'.format(backbone, dropping_rate)) + '>')
plt.close()
