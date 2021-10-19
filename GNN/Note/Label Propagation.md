## 标签传播算法（LPA）

设数据集为$X\in R^{n\times d}$

令$L\in R^{l\times d}$表示带标签数据集，$U\in R^{u\times d}$表示无标签数据集，$X=L\cup U$

一般$l\ll n$

定义概率转移矩阵$P=D^{-1}A$

其中，$A$为邻接矩阵，$D$为度对角矩阵，$D_{ii}=\sum_{j=1}^{n} w_{ij}$
$$
P_{ij}=P(i\to j)=\frac{w_{ij}}{\sum_{k=1}^{n}w_{ik}}
$$
定义矩阵$Y^{(0)}=\begin{pmatrix}y_1^{(0)}\\y_2^{(0)}\\\vdots\\y_n^{(0)}\end{pmatrix}$

对于$i\le l$，$v_i$为带标签节点，$y_i$为one-hot向量，指示$v_i$的类别

对于$l<i\le n$，$v_i$为无标签节点，$y_i$可以初始化为$0$向量



#### 标签传播

$$
\begin{cases}
Y^{(k+1)}=PY^{(k)}\\
y_i^{(k+1)}=y_i^{(0)},\forall i\le l
\end{cases}
$$

每次使用概率转移矩阵$P$对$Y$进行迭代，并固定住带标签节点对应的向量$y_i$

#### 收敛性

设$f=\begin{pmatrix}f_L\\f_U\end{pmatrix}$，初始化$f_L=Y_L^{(0)}$，$f_U=Y_U^{(0)}$

将$P$分为几个子矩阵
$$
P=\begin{pmatrix}P_{LL}&P_{LU}\\P_{UL}&P_{UU}\end{pmatrix}
$$
则$(2)$式可改写为
$$
\begin{cases}
f_U^{(k+1)}=P_{UU}f_U^{(k)}+P_{UL}f_L^{(0)}\\
f_L^{(k+1)}=f_L^{(0)}
\end{cases}
$$
显然，我们只关心$f_U$，其通项公式为
$$
f_U^n=(P_{UU})^{n}f_U^0+(\sum_{i=1}^n(P_{UU})^{i-1})P_{UL}f_L^0
$$
由于$P$的每一行都满足归一化条件，有
$$
\exist \gamma<1,\quad \sum_j^u(P_{UU})_{ij}\le\gamma,\quad \forall i\in[1,u]
$$

因此
$$
\begin{align}
\sum_j^u(P_{UU})^n_{ij}&=\sum_j^u\sum_k^u(P_{UU})^{n-1}_{ik}(P_{UU})_{kj}\\
&=\sum_k^u(P_{UU})^{n-1}_{ik}\sum_j^u(P_{UU})_{kj}\\
&\le\sum_k^u(P_{UU})^{n-1}_{ik}\gamma\\
&\le\gamma^n
\end{align}
$$
由上可得，当$n$趋于无穷时，$P_{UU}$每一行的行和趋于0，因此矩阵的每个元素趋于0，故
$$
\lim_{n\to \infty}(P_{UU})^{n}f_U^0=\mathbf{0}
$$

$$
\lim_{n\to \infty}f_U^n=(\sum_{i=1}^\infty(P_{UU})^{i-1})P_{UL}f_L^0=(I-P_{UU})^{-1}P_{UL}f_L^0
$$

因此，$f_U$收敛于
$$
f_U=(I-P_{UU})^{-1}P_{UL}f_L^0
$$
