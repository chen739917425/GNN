# LiftPool: Lifting-based Graph Pooling for Hierarchical Graph Representation Learning  
## 本文背景

常见的top-k池化一般包含两个阶段

* 节点选择：为节点分配重要性得分，选择最重要的若干节点保留，移除其余节点
* 图粗化：为保留的节点连边，获得新的拓扑结构

节点的特征耦合了结构信息和节点属性，上述过程中直接丢弃节点容易丢失结构信息

## 本文方法

本文在上述两阶段的池化中插入一个阶段，提出了三阶段的池化机制，称为**LiftPool**

### Lifting结构

将图$x$分为两个不交的子图$x_o$和$x_e$

使用一个prediction操作$P$，通过减去从$x_e$中预测得到的低频信号，来获得高频信号$\hat{x}_o$（局部信息）
$$
\hat{x}_o=x_o-Px_e
$$
然后对高频信号应用update操作$U$，并将其传播给$x_e$
$$
\hat{x}_e=x_e+U\hat{x}_o
$$
通过上述的**lifting**方法，我们可以看作把图$x$压缩到了子图$x_e$，得到了一个粗略的近似$\hat{x}_e$

**lifting**过程可以看作去除$x_o$和$x_e$的相关性

$(1)$式可以看作通过$x_e$来预测$x_o$中的全局信息，并将其从$x_o$中去除，得到$x_o$独有的局部信息$\hat{x}_o$

$(2)$式将$x_o$的局部信息送入$x_e$，生成一个更加准确的估计$\hat{x}_e$

### LiftPool

#### 节点选择

使用SAGPool中的基于注意力的方法来选择点

利用注意力得分$S$来决定每个节点的重要性
$$
S = \sigma(W_aH\Theta^s)
$$
其中$W_a$是标准化且加了自环$\lambda I$的邻接矩阵，$\Theta^s$是参数

根据池化比例$\eta$来选择被保留的节点$V^p$，$|V^p|=\eta*|V|,|V^r|=|V|-|V^p|$



#### graph lifting

在节点选择和图粗化之间插入一个lifting的阶段

记 节点选择中被保留的节点为$V^p$，舍弃的节点为$V^r$，其特征构成的矩阵分别为$H^p\in R^{|V^p|\times d}$和$H^r\in R^{|V^r|\times d}$

**lifting**之后的特征为
$$
[\hat{H}^p, \hat{H}^r] = L_{\Theta}(H^p, H^r)
$$
其中$L_{\Theta}$为上述的Lifting结构，对应参数为$\Theta$

具体地
$$
\hat{H}^r = H^r - P_{\Theta^P}(H^p)\\
\hat{H}^p = H^p + P_{\Theta^U}(\hat{H}^r)
$$
$\Theta^P$和$\Theta^U$分别是prediction操作和update操作的可学习参数

其中两种操作具体为
$$
P_{\Theta^P}(H^p) = ReLU(W_a^{pr}H^p\Theta^P)\\
P_{\Theta^U}(\hat{H}^r) = ReLU(W_a^{rp}\hat{H}^r\Theta^U)
$$
其中$W^{pr}$表示点集$V^p$指向点集$V^r$的边构成的邻接矩阵，$W^{rp}$表示点集$V^r$指向点集$V^p$的边构成的邻接矩阵

特别地，对于无向图，有$W^{pr}=W^{rp}$

将$\hat{H}^p$作为下一层节点的输入特征

#### 图粗化

保留$V^p$内部的边作为下一层图的拓扑结构
