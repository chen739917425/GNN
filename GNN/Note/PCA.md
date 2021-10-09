## 主成分分析

常用于高维数据降维

### 计算过程

设样本集为$X=\{x_1,x_2,\cdots,x_m\},X\in R^{d\times m}$

令$\mu$表示$X$的均值样本
$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x_i
$$
对每个样本进行中心化
$$
x_i=x_i-\mu
$$


计算协方差矩阵$\Sigma$
$$
\Sigma=XX^T
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

设$g(y)=Wy,W\in R^{d\times d'}$，出于简化模型的目的，限制$W$满足列向量单位正交

优化目标为
$$
\begin{align}
argmin_y||x-g(y)||_2^2&=argmin_y((x-g(y))^T(x-g(y)))\\
&=argmin_y(x^Tx-x^Tg(y)-g(y)^Tx+g(y)^Tg(y))\\
&=argmin_y(-2x^Tg(y)+g(y)^Tg(y))\\
&=argmin_y(-2x^TWy+y^TW^TWy)
\end{align}
$$

上式对$y$求偏导，令结果为$0$，可得
$$
y=W^Tx
$$
则重构得到的向量为
$$
g(y)=Wy=WW^Tx
$$
考虑整个样本集，用Frobenius范数来衡量接近程度，则目标为
$$
argmin_W||X-WW^TX||_F\\
s.t. W^TW=I
$$

##### Frobenius 范数

$$
||A||_F=\sqrt{\sum_{i,j}a_{ij}^2}=\sqrt{Tr(A^TA)}
$$

#### 求解

令$W=\{w_1,w_2,\cdots,w_{d'}\}$
$$
\begin{align}
argmin_W||X-WW^TX||_F&=argmin_WTr((X-WW^TX)^T(X-WW^TX))\\
&=argmin_WTr(X^TX-2X^TWW^TX+X^TWW^TWW^TX)\\
&=argmin_WTr(-2X^TWW^TX+X^TWW^TWW^TX)\\
&=argmin_WTr(-X^TWW^TX)\\
&=argmax_WTr(X^TWW^TX)\\
&=argmax_WTr(W^TXX^TW)\\
&=argmax_W\sum_{i=1}^{d'}(w_i^TXX^Tw_i)\\
&=argmax_W\sum_{i=1}^{d'}(w_i^T\Sigma w_i)\\
&s.t.W^TW=I
\end{align}
$$

显然$\Sigma$是实对称矩阵，进行特征分解，可得到$d$个正交的特征向量$\{v_1,v_2,\cdots,v_d\}$

由瑞利商定理可知，当$W$取$\Sigma$最大的$d'$个特征值对应的特征向量时，上述取得最大值$\lambda_1+\lambda_2+\cdots+\lambda_{d'}$

