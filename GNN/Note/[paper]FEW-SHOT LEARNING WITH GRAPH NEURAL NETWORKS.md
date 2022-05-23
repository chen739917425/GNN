# FEW-SHOT LEARNING WITH GRAPH NEURAL NETWORKS

## 问题描述

考虑从部分标注的图像集分布中采样得到的输出-输出对$(\mathcal{T}_i,Y_i)_i$，其中
$$
\mathcal{T}=\left\{
\left\{(x_1,l_1),\cdots,(x_s,l_s)  \right\},
\left\{\tilde{x}_1,\cdots,\tilde{x}_r  \right\},
\left\{\bar{x}_1,\cdots,\bar{x}_t  \right\}
\right\}\\
Y=(y_1,\cdots,y_t)\in \{1,\cdots,K\}^t\\
l_i\in\{1,\cdots, K\}\\
x_i,\tilde{x}_j,\bar{x}_j\sim \mathcal{P}_l(\mathbb{R}^N)
$$
其中

* $s$为已知标签的样本数量，$r$为未知标签的样本数量，$t$为需要预测分类的样本数量
* $K$是样本类别数
* $\mathcal{P}_l(\mathbb{R}^N)$表示$\mathbb{R}^N$上的特定种类图像的分布
* $Y_i$是$\{\bar{x}_1,\cdots,\bar{x}_t\} \in \mathcal{T}_i$的标签

给定一个训练集$\{(\mathcal{T}_i,Y_i)_i|i\in\{1,\cdots,L\}$，我们考虑经典的监督学习目标
$$
\min_{\Theta}\frac{1}{L}\sum_{i\le L}l(\Phi(\mathcal{T}_i;\Theta),Y_i)+R(\Theta)
$$
其中模型$\Phi(\mathcal{T}_i;\Theta)=p(Y|\mathcal{T})$，$R()$为正则项

本文专注于$t=1$的情况，即每个任务$\mathcal{T}$中只预测一个样本的类别

##### Few-shot Learning

$r=0,t=1,s=qK$，$K$种标签分别恰好出现了$q$次，称为为$q$-shot,$K$-way learning

##### Semi-Supervised Learning

$r>0,t=1$，模型可以使用同分布但未知标签的样本来训练提高性能

##### Active Learning

模型有能力从未知标签的样本中获取标签，本文研究主动学习可以在多大程度上提高半监督学习设置的性能

在$s+r=s_0,s\ll s_0$的情况下，使其性能匹配于$s_0$个样本的one-shot learning

## 本文工作

#### 初始化节点特征

$$
x_i^{(0)}=\phi(x_i)||l_i
$$

其中

* $\phi$是CNN
* 若样本标签已知，则$l_i$为表示标签的one-hot向量，否则为每一维为$\frac{1}{K}$的向量，$K$为类别数

#### GCN

把$\mathcal{T}$中的样本建模为全连接图$G_{\mathcal{T}}=(V,E)$

定义$\mathcal{A}$为图上的线性操作族，对于输入信号$x^{(k)}\in R^{|V|\times d_k}$，GCN层为
$$
x^{(k+1)}=\rho(\sum_{B\in\mathcal{A}}Bx^{(k)}\theta_B^{(k)})
$$
其中

* $\theta_B^{(k)}\in R^{d_k\times d_{k+1}}$，为可训练参数
* $\rho$为ReLU操作

本文选用的$\mathcal{A}=\{\tilde{A}^{(k)},I\}$
$$
\tilde{A}^{(k)}=\varphi_{\tilde{\theta}}(x_i^{(k)},x_j^{(k)})=MLP_{\tilde{\theta}}(abs(x_i^{(k)}-x_j^{(k)}))
$$
