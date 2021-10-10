## 谱聚类

### 无向带权图

将样本看作节点，距离较远的两点间边权较小，距离较近的两点间边权较大

令$w_{ij}$表示$v_i,v_j$之间的边权，若$v_i,v_j$不相邻，$w_{ij}=0$

对于任意一个点，度$d_i$定义为
$$
d_i=\sum_{j=1}^{n}w_{ij}
$$
图的度矩阵为
$$
D=\begin{pmatrix}d_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&d_n\end{pmatrix}
$$
对于一个点集$A\in V$，定义$vol(A)$
$$
vol(A)=\sum_{i \in A}d_i
$$

### 相似性度量

定义图的邻接矩阵$W\in R^{n\times n}$，其中$W_{ij}=w_{ij}$

通过样本点间的相似性度量，来构建邻接矩阵，得到描述点间相似度的图

#### $\epsilon$-近邻图

设定一个阈值$\epsilon$
$$
W_{ij}=\begin{cases}\epsilon&||v_i-v_j||_2^2\le\epsilon\\0&otherwise\end{cases}
$$
边权的信息只有$0$和$\epsilon$，构建得到的相似性图实际可看作无权图

#### k-近邻图

每个样本只和距离自身最近的$k$个样本连边，为保证邻接矩阵的对称性，有以下两种方法
$$
W_{ij}=W_{ji}=\begin{cases}exp(-\frac{||v_i-v_j||^2_2}{2\sigma^2})&v_i\in KNN(v_j) \or v_j\in KNN(v_i)\\0&otherwise\end{cases}
$$

$$
W_{ij}=W_{ji}=\begin{cases}exp(-\frac{||v_i-v_j||^2_2}{2\sigma^2})&v_i\in KNN(v_j) \and v_j\in KNN(v_i)\\0&otherwise\end{cases}
$$

#### 全连接图

$$
W_{ij}=exp(-\frac{||v_i-v_j||^2_2}{2\sigma^2})
$$

三种建图中，全连接图最为普遍，而定义边权的核函数中，高斯核函数最为普遍



### 图切割

定义两个子集$A,B\in V,A\cap B=\empty$

则切开$A,B$的代价为
$$
C(A,B)=\sum_{i\in A,j\in B}w_{ij}
$$
令$\bar{A}$表示$A$的补集，则将图分为$k$个子集$\{A_1,A_2,\cdots,A_k\}$的代价为
$$
cut(A1,A2,\cdots,A_k)=\frac{1}{2}\sum_{i=1}^kC(A_i,\bar{A_i})
$$

#### RatioCut

最小化$RatioCut$，即在最小化切图代价的同时，最大化每个簇的点个数
$$
RatioCut(A1,A2,\cdots,A_k)=\frac{1}{2}\sum_{i=1}^k\frac{C(A_i,\bar{A_i})}{|A_i|}
$$

定义$h_i=\begin{pmatrix}h_{i,1}\\h_{i,2}\\\cdots\\h_{i,n}\end{pmatrix}$
$$
h_{i,j}=\begin{cases}\frac{1}{\sqrt{|A_i|}}&if\quad v_j \in A_i\\0&otherwise\end{cases}
$$
其中，$i\in [1,n], j\in [1,k]$

$n$为节点个数，$k$为簇个数
$$
\begin{align}
h_i^TLh_i&=\frac{1}{2}\sum_{x=1}^{n}\sum_{y=1}^{n}w_{xy}(h_{ix}-h_{iy})^2\\
&=\frac{1}{2}(\sum_{x\in A_i}\sum_{y\notin A_i}w_{xy}\frac{1}{|A_i|}+\sum_{x\notin A_i}\sum_{y\in A_i}w_{xy}\frac{1}{|A_i|})\\
&=\frac{1}{2}(cut(A_i,\bar{A_i})\frac{1}{|A_i|}+cut(\bar{A_i},A_i)\frac{1}{|A_i|})\\
&=\frac{cut(A_i,\bar{A_i})}{|A_i|}
\end{align}
$$

令$H\in R^{n\times k}$表示由$k$个列向量$h_i,i\in[1,k]$构成的矩阵，则有
$$
\begin{align}
RatioCut(A1,A2,\cdots,A_k)&=\frac{1}{2}\sum_{i=1}^k\frac{C(A_i,\bar{A_i})}{|A_i|}\\
&=\sum_{i=1}^kh_i^TLh_i\\
&=\sum_{i=1}^k(H^TLH)_{ii}\\
&=tr(H^TLH)
\end{align}
$$
由于$H$是正交矩阵，有$H^TH=I$。因此优化目标等价于
$$
argmin_{H}\quad tr(H^TLH)\\
s.t.\quad H^TH=I
$$
由于该优化问题为NP，考虑近似求解

$L$是实对称矩阵，由瑞利商定理可知，$H$应取$L$最小的$k$个特征值对应的特征向量为列向量


#### Ncut

最小化$Ncut$，即在最小化切图代价的同时，最大化每个簇内的权重
$$
NCut(A1,A2,\cdots,A_k)=\frac{1}{2}\sum_{i=1}^k\frac{C(A_i,\bar{A_i})}{vol(A_i)}
$$
定义$h_i=\begin{pmatrix}h_{i,1}\\h_{i,2}\\\cdots\\h_{i,n}\end{pmatrix}$
$$
h_{i,j}=\begin{cases}\frac{1}{\sqrt{vol(A_i)}}&if\quad v_j \in A_i\\0&otherwise\end{cases}
$$
类似地，推导可得
$$
h_i^TLh_i=\frac{cut(A_i,\bar{A_i})}{vol(A_i)}
$$
令$H\in R^{n\times k}$表示由$k$个列向量$h_i,i\in[1,k]$构成的矩阵，优化目标仍然为
$$
tr(H^TLH)
$$
但此时约束为$H^TDH=I$
$$
h_i^TDh_i=\sum_{j=1}^{n}h_{ij}^2d_j=\sum_{v_j\in A_i}\frac{d_j}{vol(A_i)}=\frac{\sum_{v_j\in A_i}d_j}{vol(A_i)}=\frac{vol(A_i)}{vol(A_i)}=1
$$
因此，优化问题为
$$
argmin_H\quad tr(H^TLH)\\
s.t.\quad H^TDH=I
$$
设$F=D^{\frac{1}{2}}H$，则优化问题转为
$$
argmin_H\quad tr(F^TD^{-\frac{1}{2}}LD^{-\frac{1}{2}}F)\\
s.t.\quad F^TF=I
$$
类似地，$F$应取$L_{sym}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$最小的$k$个特征值对应的特征向量为列向量

### 谱聚类算法

输入：样本集$X\in R^{m\times n}$，分类的簇数$k$

输出：簇划分$A_1,A_2,\cdots,A_k$

* 根据相似性，计算得到邻接矩阵$W\in R^{n\times n}$

* 计算度矩阵$D$和拉普拉斯矩阵$L$（或$L_{sym}$)

* 对拉普拉斯矩阵的进行特征分解，取前$k$小的特征值对应的特征向量，构成矩阵$H\in R^{n\times k}$

* 令$y_i\in R^k$表示$H$第$i$行的向量，对$y_i,i\in[1,n]$进行标准化
* 对新样本集$Y=\{y_1,y_2,\cdots,y_n\}$进行一次传统的聚类（如K-Means)
* 得到簇的划分$A_i=\{x_j|y_j\in C_i\},i\in [1,k]$

### 优劣

#### 优点

* 基于谱图理论，能在任意形状的样本空间上聚类且收敛于全局最优解

* 对于处理稀疏数据比较有效
* 由于降维，复杂度比传统聚类优秀

#### 缺点

* 聚类效果依赖于相似性的度量方法
* 不适用于簇类别较多的聚类问题
* 不适用于各簇之间样本数量差别很大的聚类问题
