# GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING

## 问题描述

问题可以被形式化的描述为分散式不完全观测马尔科夫决策过程（Decentralized Partially Observable Markov Decision Process，**Dec-POMDP**）

在每个时间步$t$，智能体$i$获得局部观测$o_i^t$，采取的动作记为$a_i^t$，得到的个体奖励记为$r_i^t$

目标为最大化所有智能体的期望奖励之和

智能体的位置信息和数量都有可能随着时间而变化

## 本文工作

本文将多智能体（multi-agent）环境建模为一张图，每个智能体（agent）作为图上的节点，将其局部观测作为节点的属性，

每个节点$i$的邻居集合为$\mathbb{B}_i$，通过距离之类的度量来定义，取决于环境且可能随着时间而改变

使用的强化学习方法基于Q-learning方法

模型命名为DGN，主要由三个部分组成：**编码器，图卷积层和Q网络**

<img src="C:\Users\Ging\GingStudy\GNN\Note\[paper]GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING\image-20220221151850136.png" alt="image-20220221151850136" style="zoom:80%;" />

### Encoder

编码器将每个智能体$i$的观测$o_i^t$编码为特征向量$h_i^t$

对于低维输入，可以选用MLP作为编码器，对于视觉图像输入，可以选用CNN

### Convolutional layer

令$F^t\in\mathbb{R}^{N\times L}$表示时间步$t$下的节点特征矩阵，其中$N$是智能体的数量，$L$是特征向量的长度

令$C_i^t\in \mathbb{R}^{(|\mathbb{B}_i|+1)\times N}$表示时间步$t$下智能体$i$的邻域，$C_i^t$的第一行为关于节点$i$的索引的one-hot向量，第$j\in[2,|\mathbb{B}_i|+1]$行表示关于节点$i$的第$j$个邻点的索引的ont-hot向量

我们可以通过$C_i^t\times F_t$来获取智能体$i$的局部特征向量

#### Relation kernel

令$\mathbb{B}_{+i}$表示智能体$i$本身与其邻居节点集合$\mathbb{B}_i$的并

使用多头自注意力机制，对于注意力头$m$，$i$和$j\in \mathbb{B}_{+i}$的注意力系数如下计算
$$
\alpha^m_{i,j}=\frac{\exp\left(\tau\cdot W_Q^m h_i\cdot(W_K^mh_j)^T\right)}{\sum_{k\in\mathbb{B_{+i}}}\exp\left(\tau\cdot W^m_Qh_i\cdot(W^m_Kh_k)^T\right)}
$$
其中$\tau$是一个放缩因子

对于每个智能体$i$，将其邻域节点的value表示用注意力系数加权求和，最后将$M$头注意力下的结果拼接，得到卷积层的输出$h'_i$
$$
h'_i=\sigma\left(||_{m\in M}\sum_{j\in\mathbb{B}_{+i}}\alpha_{i,j}^mW_V^mh_j\right)
$$
其中$\sigma$可以是一个单层MLP套一个非线性激活函数ReLU

可以堆叠上述卷积层来提取更高层的关系表示，然后将每一层的输出结果拼接后送入Q network中作为其输入

![image-20220221163230915](C:\Users\Ging\GingStudy\GNN\Note\[paper]GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING\image-20220221163230915.png)

### Q network

本文实验中使用的Q network使用的是仿射变换

### 训练

在训练过程中，每个时间步下，我们储存元组$(\mathcal{O},\mathcal{A},\mathcal{O}',\mathcal{R},\mathcal{C})$到回放池中

其中$\mathcal{O}=\{o_1,o_2,\cdots,o_N\}$是每个智能体观测的集合，$\mathcal{A}=\{a_1,a_2,\cdots,a_N\}$是智能体对应于观测做出的动作的集合，$\mathcal{O}'=\{o'_1,o'_2,\cdots,o'_N\}$是智能体做出动作后进入的下一个状态的集合，$\mathcal{C}=\{C_1,C_2,\cdots,C_N\}$是每个智能体的邻接矩阵

然后我们从回放池中取出一个大小为$S$的minibatch，最小化如下损失
$$
\mathcal{L}(\theta)=\frac{1}{S}\sum_{S}\frac{1}{N}\sum_{i=1}^{N}(y_i-Q(O_{i,C},a_i;\theta))^2
$$
其中$y_i=r_i+\gamma\max_{a'}Q(O'_{i,C},a'_i;\theta')$，$O_{i,C}\subseteq \mathcal{O}$表示智能体$i$由$\mathcal{C}$定义的邻域得到的观测的集合

由于图一直在改变，不利于Q网络的学习，因此在训练中我们固定$\mathcal{C}$在两个连续的时间步内不变

对于target network的参数如下更新
$$
\theta'=\beta\theta+(1-\beta)\theta'
$$

### Temporal relation regularization

本文认为协作关系应该是长期一致的，即使图一直在改变

为了使注意力权重保持稳定，提出了时间相关的关系正则项

即希望当前时间步下的注意力权重分布与下一时间步下的注意力权重分布尽量接近

此外本文还认为越高层捕获到的关系表示应该越抽象且稳定，因此将上述正则项用于网络的高层

定义$\mathcal{G}_m^k(O_{i,C};\theta)$表示智能体$i$在注意力头$m$下第$k$层卷积层的注意力权重分布，其中$k$为超参，设定要应用正则项的卷积层

最终的损失函数如下
$$
\mathcal{L}(\theta)=\frac{1}{S}\sum_{S}\frac{1}{N}\sum_{i=1}^{N}(y_i-Q(O_{i,C},a_i;\theta))^2+\lambda\frac{1}{M}\sum_{m=1}^MD_{KL}(\mathcal{G}_m^k(O_{i,C};\theta)||\mathcal{G}_m^k(O'_{i,C};\theta))
$$

## 实验

### 模型

作者设置了以下模型与DGN作对比

* DGN-R：将DGN的时间关系正则项移除
* DGN-M：用均值操作代替注意力机制
* MFQ：平均场Q-learning
* DQN：deep Q network
* CommNet：Communication Neural Net，交流神经网

### 环境

#### MAgenta

$30\times 30$的网格世界

每个智能体可以观测到以自己为中心，$11\times 11$正方形区域

有两种场景：battle和jungle

#### packet switching network

模拟在分组交换网络上路由（routing）

![image-20220222141553247](C:\Users\Ging\GingStudy\GNN\Note\[paper]GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING\image-20220222141553247.png)

### 结果

#### Battle

$N=20,L=12,2000$个episode

![image-20220222142900131](C:\Users\Ging\GingStudy\GNN\Note\[paper]GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING\image-20220222142900131.png)

#### Jungle

$N=20,L=12,2000$个episode

![image-20220222142936835](C:\Users\Ging\GingStudy\GNN\Note\[paper]GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING\image-20220222142936835.png)

#### Routing

![image-20220222143007772](C:\Users\Ging\GingStudy\GNN\Note\[paper]GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING\image-20220222143007772.png)

