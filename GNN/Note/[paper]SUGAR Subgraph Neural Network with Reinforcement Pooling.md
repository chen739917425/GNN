# SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism

## 问题描述

定义图为$G=(V,X,A),V=\{v_1,\cdots,v_N\},X\in R^{N\times d},A\in R^{N\times N}$

给定数据集$(\mathcal{G},\mathcal{Y})=\{(G_1,y_1),\cdots,(G_n,y_n)\}$

学习映射$f:\mathcal{G}\to\mathcal{Y}$进行图分类任务

## 本文工作

* 引入subgraph neural network
* 使用强化学习池化机制
* 使用自监督的互信息机制

### 子图采样与编码

将所有节点按照度数降序排列

选取前$n$个节点作为中心节点

对于每个中心节点，使用**BFS**进行扩展

由此我们得到$n$个子图$\{g_1,\cdots,g_n\}$，每个子图的节点数固定为$s$



我们学习一个基于GNN的编码器$\mathcal{E}:R^{s\times d}\times R^{s\times s}\to R^{s\times d_1}$，以获取子图内的节点表示

对于子图$g_i$
$$
H(g_i)=\mathcal{E}(g_i)=\{h_j|v_j\in V(g_i)\}
$$
使用消息传递框架来建模$\mathcal{E}$
$$
h_i^{(l+1)}=U^{(l+1)}\left(h_i^{(l)},AGG\left(M^{(l+1)}(h_i^{(l)},h_j^{(l)})|v_j\in N(v_i)\right)\right)
$$
其中$M$是消息生成函数，$AGG$是聚合函数，$U$是状态更新函数

各种GNNs的变体也可以用于$(1)$



接着我们使用子图内部注意力机制学习子图内节点的重要性

对于节点$v_j$的注意力系数$c_j^{(i)}$
$$
c_j^{(i)}=\sigma(a^T_{intra}W_{intra}h_j^{(i)})
$$
其中$a_{intra}\in R^{d_1}$是权重向量，$W_{intra}\in R^{d_1\times d_1}$是权重矩阵，两者对所有子图共享

再对注意力系数做**softmax**归一化

最后得到子图$g_i$的表示$z_i$
$$
z_i=\sum_{v_j\in V(g_i)}c_j^{(i)}h_j^{(i)}
$$

### 子图选择

使用池化比率$k\in(0,1]$自适应的**top-k**采样来挑选一部分子图

使用一个可训练的向量$p$将所有的子图的表示投影到一维$\{val_i|g_i\in G\}$
$$
val_i=\frac{z_ip}{||p||}
$$
$val_i$度量子图$g_i$投影到$p$方向上时保留多少信息

使用$\{val_i\}$作为子图的重要程度并降序排列

我们保留前$n'=\lceil k\cdot n\rceil$个子图，其余的舍弃
$$
idx=rank(\{val_i\},n')
$$

#### 强化学习池化模型

参数$k$使用强化学习来学习

定义马尔科夫决策过程

* 状态：$s_e$表示epoch $e$下的状态
  $$
  s_e=idx_e
  $$

* 动作：定义动作$a$表示从$k$上加上或减去一个固定的值$\Delta k$

* 转移：在$k$更新后，在下一个epoch使用$(6)$选择新的子图集合，得到下一个转移到的状态

* 奖励：使用离散的奖励函数，基于分类结果来给出奖励
  $$
  r(s_e,a_e)=\begin{cases}
  +1&if\,acc_e>acc_{e-1}\\
  0&if\,acc_e=acc_{e-1}\\
  -1&if\,acc_e<acc_{e-1}\\
  \end{cases}
  $$
  其中$acc_e$是epoch $e$的分类准确率

* 终止：如果连续$10$个epoch的$k$的变化幅度没有超过$\Delta k$，则认为模型已经找到最优的$k$值，再接下来的训练中将$k$固定
  $$
  Range(\{k_{e-10},\cdots,k_e\})\le \Delta k
  $$

* 

由于这是一个离散优化问题，本文使用Q-learning方法来学习
$$
Q^*(s_e,a_e)=r(s_e,a_e)+\gamma\arg\max_{a'}Q^*(s_{e+1},a')
$$
其中$\gamma\in [0,1]$

使用$\epsilon-greedy$策略
$$
\pi(s_e;Q^*)=\begin{cases}
random\,action&pr<\epsilon\\
\arg\max_{a_e}Q^*(s_e,a_e)&otherwise
\end{cases}
$$

### 子图sketching

将子图视为超级节点，构建出一张sketched graph $G^{ske}=(V^{ske},E^{ske})$，其上的连边关系根据子图间共有节点的数量决定
$$
V^{ske}=\{g_i\},\forall i\in idx\\
E^{ske}=\{e_{i,j}\},\forall |V(g_i)\cap V(g_j)|>b_{com}
$$
$b_{com}$是预定义的阈值

使用子图间多头注意力自适应地学习超级节点（子图）间的相互影响，计算新图上超级节点的embedding
$$
z'_i=\frac{1}{M}\sum_{m=1}^M\sum_{e_{ij}\in E^{ske}}a_{ij}^mW_{inter}^mz_j
$$
其中

* $a_{ij}$表示子图$g_i,g_j$间的注意力系数，可以用GAT中的方法计算得到
  $$
  a_{ij}=a(W^m_{inter}z_i,W^m_{inter}z_j)
  $$
  其中$a(\cdot)$是计算两个节点相似性的函数

* $W_{inter}\in R^{d_2\times d_1}$是一个权重矩阵
* $M$是多头注意力的头数

#### 自监督互信息模型

本文使用自监督互信息机制来进一步强化子图embedding

使用一个readout函数获取全图的表示
$$
r=READOUT(\{z'_i\}_{i=1}^{n'})
$$
其中readout可以是任意具有排列不变性的函数，如均值操作，图级别池化等，本文选用的是简单的均值

使用**Jensen-Shannon (JS) MI estimator**来最大化子图embedding与全局图embedding的互信息

具体地，引入判别器$\mathcal{D}：R^{d_2}\times R^{d_2}\to R$

我们使用双线性函数作为判别器
$$
\mathcal{D}(z'_i,r)=\sigma(z'_iW_{MI}r)
$$
其中$W_{MI}$是得分矩阵，$\sigma(\cdot)$是sigmoid函数

采用对比学习的方式，采样batch中的其他图$\tilde{G}$作为负样本

计算图$\tilde{G}$的子图embedding $\tilde{z}'$

自监督互信息的优化目标为
$$
\mathcal{L}_{MI}^{G}=\frac{1}{n'+n_{neg}}\left(\sum_{g_i\in G}^{n'}\mathbb{E}_{pos}\left[\log\left(\mathcal{D}(z'_i,r)\right)\right]+\sum_{g_j\in \tilde{G}}^{n_{neg}}\mathbb{E}_{neg}\left[\log\left(1-\mathcal{D}(\tilde{z}'_j,r)\right)\right]\right)
$$


### 预测

在得到新图上超级节点的embedding后，我们使用**softmax**将其转换为标签预测

最后，我们将所有子图的分类结果求和作为整张图最终的概率分布

### 优化

整个模型SUGAR的最终优化目标如下
$$
\mathcal{L}=\mathcal{L}_{classify}+\beta\sum_{G\in\mathcal{G}}\mathcal{L}_{MI}^{G}+\lambda||\Theta||^2
$$
其中$\mathcal{L}_{classify}$是分类的交叉熵损失，$\Theta$是模型的可训练参数