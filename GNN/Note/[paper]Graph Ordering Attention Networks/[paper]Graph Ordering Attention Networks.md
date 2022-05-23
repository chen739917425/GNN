

# Graph Ordering Attention Networks

## 背景

由于邻域节点没有顺序关系，以往的GNN使用的聚集操作要满足排列不变性，如$max(), mean() ,sum()$等

这些排列不变的聚合操作，忽略了邻域节点间的潜在协同关系，把每个邻域节点看作独立的

## 前置

### 符号

$\bar{\mathcal{N}}(u)=\mathcal{N}(u)\cup\{u\}$，即邻域节点并上节点自身

$H_{\bar{\mathcal{N}}(u)}=\{h_v|v\in\bar{\mathcal{N}}(u)\}$

### 概念

$\pi$为集合$S_M$的任意一个排列

##### 排列不变性

$$
f(\{x_1,x_2,\cdots,x_M\})=f(\{x_{\pi(1)},x_{\pi(2)},\cdots,x_{\pi(M)}\})
$$

##### 排列等变性

$$
\pi f(\{x_1,x_2,\cdots,x_M\})=f(\{x_{\pi(1)},x_{\pi(2)},\cdots,x_{\pi(M)}\})
$$

### 图视角下的信息论

#### Partial Information Decomposition

对于$u\in V$，$h_u$与$H_{\bar{\mathcal{N}}(u)}$的互信息为
$$
I(h_u;H_{\bar{\mathcal{N}}(u)})=\int\int p(h_u;H_{\bar{\mathcal{N}}(u)})\log(\frac{p(h_u,H_{\bar{\mathcal{N}}(u)})}{p(h_u)p(H_{\bar{\mathcal{N}}(u)})})d\,h_u\,d\,H_{\bar{\mathcal{N}}(u)}
$$
可以拆解为
$$
I(h_u;H_{\bar{\mathcal{N}}(u)})=\sum_{v\in\bar{N}(u)}U_v+R+S
$$


<img src="C:\Users\Ging\GingStudy\GNN\Note\[paper]Graph Ordering Attention Networks\image-20220424172347021.png" alt="image-20220424172347021" style="zoom:75%;" />

其中

* $U_v$表示节点$v$独有的信息

* $R$表示各节点所重叠的信息

* $S$表示剩下的信息，为协同信息，由邻域节点的组合来共同表现，需要考虑节点间的相互作用来捕获

#### 聚集操作捕获的信息

传统的聚集操作，得到的信息为$\sum_{v\in N(u)}I(h_u;h_v)$，缺少了协同信息$S$

## 本文工作

本文提出GOAT(Graph Ordering Attention)层结构

![image-20220425001416853](C:\Users\Ging\GingStudy\GNN\Note\[paper]Graph Ordering Attention Networks\image-20220425001416853.png)

#### Ordering Part

为无序的邻域节点集合学习一个排列顺序

对于节点$v_i\in V$，其邻域节点$v_j$得分为
$$
a_{ij}=LeakyReLU(w_2^T[W_1h_i||W_1h_j])
$$
其中$W_1\in R^{d'\times d'},w_2\in R^{2d'}$

将邻域节点按得分为关键字排序，得到排列$\pi$
$$
h_{sorted(i)}=[h_{\pi(1)},h_{\pi(2)},\cdots,h_{\pi(|\bar{\mathcal{N}}(u)|)}]
$$

#### Sequence Modeling Part

将得到的邻域节点有序序列输入到RNN网络中，来获得节点$v_i$的表示
$$
h_i^{new}=LSTM(h_{sorted(i)})
$$
本文使用的是LSTM，作者认为LSTM的遗忘门利于丢弃重叠信息，记忆门可以识别协同信息，输入门可以分离出独有信息

#### Multi-head Attention Ordering

使用多头注意力获得多种邻域节点的排列$h^k_{sorted(i)}$

输入到$K$个LSTM中分别得到输出并拼接
$$
h_i^{new}=||_{k=1}^KLSTM^k(h^k_{sorted(i)})
$$

#### 排列等变性和单射性

显然GOAT具有排列等变性

可以证明，在理论上GOAT可以任意近似单射函数
