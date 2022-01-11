# RandomDropping

We unify most of existing random dropping methods on GNNs, which includes Dropout[1], DropEdge[2], and DropNode[3].

Besides, we propose a random dropping method directly on the message matrix, which we call DropMessage.

### Installation

We used the following Python packages for core development.

```
pytorch >= 1.8.0
torch-cluster             1.5.9              
torch-geometric           2.0.2
torch-scatter             2.0.9 
torch-sparse              0.6.12
torch-spline-conv         1.2.1
```

### Dataset download

All the necessary data files can be downloaded from the PyG.

### Run

To conduct experiments, we can run the `train.py` for simple training, and run the `XX_average_train.py` to select hyperparameters and train multiple rounds.

 

----

[1] Geoffrey E. Hinton,  Nitish Srivastava,A. Krizhevsky, Ilya Sutskever, and R. Salakhutdinov.  Im-proving  neural  networks  by  preventing  co-adaptation  offeature detectors.ArXiv, abs/1207.0580, 2012.

[2] Yu Rong, Wenbing Huang, Tingyang Xu,and Junzhou Huang.  Dropedge: Towards deep graph con-volutional networks on node classification. InICLR, 2019.

[3] Wenzheng   Feng,    Jie   Zhang,    YuxiaoDong,  Yu  Han,  Huanbo  Luan,  Qian  Xu,  Qiang  Yang,Evgeny  Kharlamov,  and  Jie  Tang.    Graph  random  neural  networks  for  semi-supervised  learning  on  graphs.NeurIPS, 33, 2020.