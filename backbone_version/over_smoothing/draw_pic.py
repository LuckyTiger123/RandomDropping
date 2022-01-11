import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rc('font', family='Times New Roman')

pic_id = 2

backbone = 'GCN'
dropping_method_list = ['Clean', 'Dropout', 'DropEdge', 'DropNode', 'DropMessage']

x = range(1, 9)
fig, ax = plt.subplots()
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)

# ax.plot(x, [0.3550, 0.3928, 0.4005, 0.3838, 0.3598, 0.3289, 0.2704, 0.2185],
#         label='GCN' + '-' + dropping_method_list[0])
# ax.plot(x, [0.3585, 0.4231, 0.4247, 0.4126, 0.3998, 0.3869, 0.3558, 0.3439],
#         label='GCN' + '-' + dropping_method_list[1])
# ax.plot(x, [0.3588, 0.4096, 0.4158, 0.3988, 0.3957, 0.3854, 0.3679, 0.3572],
#         label='GCN' + '-' + dropping_method_list[2])
# ax.plot(x, [0.3581, 0.4225, 0.4346, 0.4107, 0.3928, 0.3780, 0.3572, 0.3503],
#         label='GCN' + '-' + dropping_method_list[3])
# ax.plot(x, [0.3588, 0.4233, 0.4397, 0.4165, 0.4079, 0.3905, 0.3774, 0.3654],
#         label='GCN' + '-' + dropping_method_list[4])
# ax.set_xlabel('Layer', fontsize=32)
# ax.set_ylabel('MADGap', fontsize=32)

ax.plot(x, [77.08, 80.52, 77.86, 70.86, 60.62, 43.48, 36.57, 33.08],
        label='GCN' + '-' + dropping_method_list[0])
ax.plot(x, [77.74, 81.45, 78.65, 73.49, 65.41, 52.59, 44.10, 38.09],
        label='GCN' + '-' + dropping_method_list[1])
ax.plot(x, [77.20, 81.30, 78.87, 73.53, 68.09, 56.46, 45.87, 40.88],
        label='GCN' + '-' + dropping_method_list[2])
ax.plot(x, [77.85, 81.42, 78.73, 73.71, 63.59, 53.99, 42.11, 37.27],
        label='GCN' + '-' + dropping_method_list[3])
ax.plot(x, [77.91, 81.82, 79.35, 73.85, 68.26, 58.46, 46.57, 42.74],
        label='GCN' + '-' + dropping_method_list[4])
ax.set_xlabel('Layer', fontsize=32)
ax.set_ylabel('Accuracy(%)', fontsize=32)

# ax.set_title('Test loss on {} dataset'.format(dataset))
# ax.legend(fontsize=16)

plt.show()

fig.savefig('./PIC/pic_{}_nl.pdf'.format(pic_id), dpi=600, format='pdf')
