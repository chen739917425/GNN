## 主成分分析

常用于高维数据降维

### 计算过程

设样本集为$X=\{x_1,x_2,\cdots,x_m\},X\in R^{d\times m}$

令$\mu$表示$X$的均值样本
$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x_i
$$
计算协方差矩阵$\Sigma$
$$
\Sigma=(x-\mu)(x-\mu)^T
$$
对$\Sigma$进行特征分解

选取前$d'$大的特征值对应的特征向量$\{v_1,v_2,\cdots,v_{d'}\}$

令投影矩阵$W\in R^{d\times d'}$为
$$
W=\begin{pmatrix}v1,v2,\cdots,v_{d'}\end{pmatrix}
$$
则可得到降维后的样本集$X'\in R^{d' \times m}$
$$
X'=W^TX
$$

### 原理推导

#### 目标

将$d$维空间中的向量$x$映射到$d'$维空间中的向量$y$（编码），同时考虑重构（解码）损失

设$g(y)=Wy$，出于简化模型的目的，限制$W$满足列单位正交

优化目标为
$$
argmin_y||x-g(y)||_2^2
$$




