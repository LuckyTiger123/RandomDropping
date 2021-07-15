import os
import pandas as pd
import matplotlib.pyplot as plt


def get_xdata(filename, dropping_method):
    df = pd.read_excel(filename.format(dropping_method))
    dd = df.to_numpy()
    dd = dd.transpose()
    xx = dd[1]  # epoch
    return xx

def get_ydata(filename, dropping_method):
    print('open <' + filename.format(dropping_method) + '> !')

    df = pd.read_excel(filename.format(dropping_method))
    dd = df.to_numpy()
    dd = dd.transpose()
    yy = dd[2]  # loss
    return yy


dataset = 'Cora' # ['Cora', 'CiteSeer', 'PubMed']
backbone = 'APPNP' # ['GAT', 'GCN', 'APPNP']
# dropping_rate = '0.1'

for dataset in ['Cora', 'CiteSeer', 'PubMed']:
    for backbone in ['GAT', 'GCN', 'APPNP']:
        for dropping_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            filepath = './result/' + dataset + '/'
            filename = filepath + backbone + '_{}_' + str(dropping_rate) + '_lossfunc.xlsx'

            dropping_method_list = ['DropNode', 'DropEdge', 'Dropout', 'DropMessage']
            xx = get_xdata(filename, dropping_method_list[0])
            yy_list = []
            yy_list.append(get_ydata(filename, dropping_method_list[0]))
            yy_list.append(get_ydata(filename, dropping_method_list[1]))
            yy_list.append(get_ydata(filename, dropping_method_list[2]))
            yy_list.append(get_ydata(filename, dropping_method_list[3]))

            fig, ax = plt.subplots()
            ax.plot(xx, yy_list[0], label=backbone + '-' + dropping_method_list[0])
            ax.plot(xx, yy_list[1], label=backbone + '-' + dropping_method_list[1])
            ax.plot(xx, yy_list[2], label=backbone + '-' + dropping_method_list[2])
            ax.plot(xx, yy_list[3], label=backbone + '-' + dropping_method_list[3])
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('Training loss on {} dataset'.format(dataset))
            ax.legend()
            #fig.show()

            savepath = './fig_loss/' + dataset + '/'
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            fig.savefig(os.path.join(savepath, '{}_{}_lossfunc.svg'.format(backbone, dropping_rate)))
            print('save <' + os.path.join(savepath, '{}_{}_lossfunc.svg'.format(backbone, dropping_rate)) + '>')
            plt.close()

