# Unsupervised Differentiable Multi-aspect Network Embedding

## 预备知识

#### 基于随机游走的无监督的图Embedding

受到基于**skip-gram**的**word2vec**的启发，先前的图表示学习方法将图上的node看作文本中的word，将随机游走得到的路径类比为句子

对于节点表示的学习，可以通过最大化**skip-gram**目标
$$
\max\sum_{w\in\mathcal{W}}\sum_{v_i\in w}\sum_{v_j\in N(v_i)}\log\frac{\exp(\langle P_i,Q_j\rangle )}{\sum_{v_j'\in \mathcal{V}}\exp(\langle P_i,Q_{j'}\rangle )}
$$
其中

* $P_i\in R^d$，是节点$v_i$的center embedding，$Q_j\in R^d$是节点$v_j$的context embedding，$d$为embedding的维数
* $\langle \cdot,\cdot\rangle $表示内积
* $N(v_i)$表示$v_i$的上下文窗口中的节点
* $\mathcal{W}$为随机游走得到的路径集合，$w$表示一条随机游走的路径

最大化上述目标，会使得共同出现在一个上下文窗口中的节点具有相似的embedding

该方法受限于，每个节点被一个单独的embedding表示

## 问题描述

定义图$\mathcal{G}=(\mathcal{V},\mathcal{E})$，我们希望对于每个节点$v_i$得到一个multi-aspect的embedding矩阵$Q_i\in R^{K\times d}$

其中，$K$是预先设定的任意大小的aspect的数量

每个aspect的embedding向量$\{Q^{(s)}_i\in R^d\}_{s=1}^K$应该

* 保留图的结构信息
* 捕捉节点$v_i$的不同aspect
* 建模不同aspect之间的相互影响

## 本文工作

### 基于上下文的Multi-aspect的图Embedding

给定一个目标节点的embedding $P_i\in R^d$ 和它目前选择的aspect $\delta(v_i)\in\{1,2,\cdots,K\}$，我们的目标是预测它的context embedding $\{Q_j^{\delta(v_i)}|v_j\in N(v_i)\}$

对于每一条随机游走序列$w$，我们最大化目标
$$
\mathcal{J}^{(w)}_{asp2vec}=\sum_{v_i\in w}\sum_{v_j\in N(v_i)}\sum_{s=1}^Kp(\delta(v_i)=s)\log p(v_j|v_i,p(\delta(v_i)=s))
=\sum_{v_i\in w}\sum_{v_j\in N(v_i)}\sum_{s=1}^Kp(\delta(v_i)=s)\log\frac{\exp(\langle P_i,Q_j^{(s)}\rangle )}{\sum_{v_j'\in\mathcal{V}}\exp(\langle P_i,Q_{j'}^{(s)}\rangle )}
$$

### 确定中心节点的Aspect

我们假定每个节点$v_i$的aspect可以通过检查它的上下文$N(v_i)$确定，即$p(\delta(v_i))\equiv p(\delta(v_i)|N(v_i))$。我们使用softmax建模$v_i$的aspect的概率
$$
p(\delta(v_i)=s)\equiv p(\delta(v_i)=s|N(v_i))=softmax(\langle P_i,\mathrm{Readout}^{(s)}(N(v_i))\rangle )
=\frac{\exp \left( \langle P_i,\mathrm{Readout}^{(s)}(N(v_i))\rangle\right)}{\sum_{s'=1}^K\exp\left(\langle P_i,\mathrm{Readout}^{(s')}(N(v_i))\rangle\right)}
$$


#### 基于Gumbel-Softmax选择Aspect

基于softmax来建模aspect的选择概率$p(\delta(v_i))$​，得到的是一个连续的概率分布。可是我们认为，虽然一个节点在全局角度下可以同时属于多个aspect，但在一个局部上下文中，它应该属于一个单独的aspect，即硬选择。然而硬选择是一个不可微的操作，会使得梯度的传播中断，导致训练无法进行。

因此，我们采用**Gumbel-Softmax**技巧，近似以概率分布进行采样。

我们假定有一个$K$类的概率分布，$K$个类的概率分别是$\pi_1,\pi_2,\cdots,\pi_K$

**gumbel-max**技巧提供了一种简单的方式，来刻画一个**one-hot**采样$z=(z1,z2,\cdots,z_K)\in R^K$，$z$中$1$的位置服从类别的分布概率
$$
z=\mathrm{one\_hot}(\mathop{\arg\max}_i[\log\pi_i+g_i])
$$
其中$g_i$是从$Gumbel(0,1)$分布中提取的**Gumbel**噪声，可以如下获得
$$
g_i=-\log(-\log(u_i)),u_i\sim Uniform(0,1)
$$
由于$\arg\max(\cdot)$也是一个不可微的操作，因此我们使用softmax来近似，即**Gumbel-Softmax**技巧
$$
z_i=\mathrm{softmax}[\log\pi_i+g_i]=\frac{\exp((\log\pi_i+g_i)/\tau)}{\sum_{j=1}^K\exp((\log\pi_j+g_j)/\tau)},i\in[1,K]
$$
其中$\tau$是一个温度参数，来控制输出近似$\arg\max(\cdot)$的程度。当$\tau\to0$时，采样$z$变为**one-hot**

我们使用**Gumbel-Softmax**替换$(3)$中的softmax，得到如下aspect的选择概率
$$
p(\delta(v_i)=s)\equiv p(\delta(v_i)=s|N(v_i))
=\frac{\exp[(\log\langle P_i,\mathrm{Readout}^{(s)}(N(v_i))\rangle+g_s)/\tau]}
{\sum_{s'=1}^K\exp[(\log\langle P_i,\mathrm{Readout}^{(s')}(N(v_i))\rangle+g_{s'})/\tau ]}
$$
为了学习基于上下文的multi-aspect的节点表示，给出要最小化的损失函数，
$$
\mathcal{L}_{asp2vec}=-\sum_{w\in\mathcal{W}}\mathcal{J}_{asp2vec}^{(w)}
$$

#### Readout函数

我们希望$\mathrm{Readout}^{(s)}(N(v_i))$使用$v_i$的上下文信息$N(v_i)$来捕获$v_i$当前的aspect $s$

作者选择了均值池化操作
$$
\mathrm{Readout}^{(s)}(N(v_i))=\frac{1}{|N(v_i)|}\sum_{v_j\in N(v_i)}Q_j^{(s)}=\bar{Q}^{(s)}_{N(v_i)}
$$
我们也可以使用其他方法来给每个上下文节点赋予不同的权重，例如基于RNN或者基于注意力机制的池化。但出于简化模型与提高计算效率的考虑，作者还是选择了均值池化

### 建模Aspect间的联系

作者认为，aspect embedding不仅应该体现多样性以捕获各个aspect的固有属性，还应该体现某种程度的相关性以捕获aspect之间共享的公共信息

为此，作者引入了一个新型的aspect正则化框架，称为$reg_{asp}$
$$
reg_{asp}=\sum_{i=1}^{K-1}\sum_{j=i+1}^{K}A\_sim(Q_*^{(i)},Q_*^{(j)})
$$
其中

* $Q_*^{(i)}\in R^{n\times d}$表示所有节点$v\in\mathcal{V}$在aspect $i$上的embedding构成的矩阵，$|\mathcal{V}|=n$
* $A\_sim(\cdot,\cdot)$衡量两个aspect embedding矩阵的相似性得分。

$A\_sim(Q_*^{(i)},Q_*^{(j)})$越大意味着两个aspect共享越多的共同信息，反之则意味着两个aspect捕获的是不同的信息。可以通过对所有节点的aspect $i$和$j$的embedding相似度求和得到
$$
A\_sim(Q_*^{(i)},Q_*^{(j)})=\sum_{h=1}^{|\mathcal{V}|}f(Q_h^{(i)},Q_h^{(j)})
$$
其中$Q_h^{(i)}$表示节点关于aspect $i$的embedding，$f(Q_h^{(i)},Q_h^{(j)})$衡量节点关于aspect $i$和$j$的两个embedding的相似性得分，其具体形式如下

$$
f(Q_h^{(i)},Q_h^{(j)})=\frac{\langle Q_h^{(i)},Q_h^{(i)}\rangle}{||Q_h^{(i)}||\,||Q_h^{(i)}||},-1\le f(Q_h^{(i)},Q_h^{(j)})\le1
$$
由于aspect之间不是完全独立的，我们应该对它们的相互关系进行建模。

为此，作者引入了一个二进制掩码$w_{i,j}^h$，根据两个aspect间的相似性得分有选择性地惩罚它们。更为具体地，先定义二进制掩码为
$$
w_{i,j}^h=\begin{cases}&1,&|f(Q_h^{(i)},Q_h^{(j)})|\ge\epsilon\\&0,&\mathrm{otherwise}\end{cases}
$$
其中$\epsilon$是一个阈值参数，它控制给定节点的一对aspect embedding之间共享的信息量。

设置较大的$\epsilon$将鼓励aspect embedding彼此相关，而小的$\epsilon$则鼓励aspect的多样性。

将$w$应用到$A\_sim(\cdot,\cdot)$中，得到
$$
A\_sim(Q_*^{(i)},Q_*^{(j)})=\sum_{h=1}^{|\mathcal{V}|}w_{i,j}^h|f(Q_h^{(i)},Q_h^{(j)})|
$$
结合asp2vec的损失函数和aspect正则损失，得到最终的损失函数
$$
\mathcal{L}=\mathcal{L}_{asp2vec}+\lambda\,reg_{asp}
$$

### 连边预测任务

每个节点有一个center embedding $P_i\in R^d$和$K$个aspect embedding $\{Q_i^{(s)}\in R^d\}_{s=0}^{K}$，共$K+1$个embedding

如下获得每个节点$v_i$的最终embedding
$$
U_i=P_i+\frac{1}{K}\sum_{s=1}^KQ_i^{(s)}
$$

### 拓展到异构图

作者使用meta-path引导的随机游走，并通过下式合并路径中的异质性

$$
\mathcal{J}^{(w)}_{asp2vec\_het}=\sum_{v_i\in w}\sum_{t\in\mathcal{T}_{\mathcal{V}}}\sum_{v_j\in N_t(v_i)}\sum_{s=1}^Kp(\delta(v_i)=s)\log p(v_j|v_i,p(\delta(v_i)=s))
$$
上式针对的是目标节点的aspect不易从本身推断的情况

对于目标节点的aspect可以从其自身信息轻松推断的情况，我们使用仅考虑单aspect的方法来简单建模，如metapath2vec方法：
$$
\mathcal{J}_{HetDW}^{(w)}=\sum_{v_i\in w}\sum_{t\in\mathcal{T}_{\mathcal{V}}}\sum_{v_j\in N_t(v_i)}\log p(v_j|v_i)
$$
最终的损失函数为
$$
\mathcal{L}_{Het}=-\sum_{\mathcal{P}\in\mathcal{S}(\mathcal{P})}\sum_{w\in\mathcal{W}_\mathcal{P}}(\mathcal{J}^{(w)}_{asp2vec\_het}+\mathcal{J}_{HetDW}^{(w)})+\lambda\,reg_{asp}
$$
其中$\mathcal{S}(\mathcal{P})$表示所有预先定义的meta-path模式
