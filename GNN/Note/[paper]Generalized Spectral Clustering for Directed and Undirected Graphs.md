# Generalized Spectral Clustering for Directed and Undirected Graphs

## 背景

传统的谱聚类一般用于无向图

作者提出一种适用于正边权的有向图的聚类方法

## 前置

### 概念

给定图$\mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{w}),N=|\mathcal{V}|$

边$(x,y)$表示从$x$到$y$的有向边



对于任意函数$\mathcal{v}:\mathcal{V}\to R_+$可以看作正的节点度量，同理任意函数$q:\mathcal{E}\to R_+$可以看作正的边度量



边权函数为$\mathcal{w}:\mathcal{V}\times\mathcal{V}\to R_+$，对于边$(x,y)\in\mathcal{E}$，$\mathcal{w}(x,y)\ge 0$，否则$\mathcal{w}(x,y)=0$

图的带权邻接矩阵为$W=\{w_{ij}\}_{i,j=1}^N\in R^{N\times N}_+$



定义图函数$f$，将图上的点映射为$n$维的复数列向量
$$
f=[f(x)]^T_{x\in\mathcal{V}}\in C^n
$$

假设图函数定义在希尔伯特空间$l^2(\mathcal{V},\mathcal{v})$（二阶可积）

则对于任意函数$f,g\in l^2(\mathcal{V},\mathcal{v})$，其内积为
$$
\left<f,g\right>_v=\sum_{x\in \mathcal{V}}\bar{f(x)}g(x)\mathcal{v}(x)
$$
其中是$\bar{f(x)}$是$f(x)$的共轭



考虑有向带权图上的随机游走$\mathcal{X}=(X_t)_{t\ge0}$

其转移矩阵为$P=[p(x,y)]_{x,y\in \mathcal{V}}$
$$
p(x,y)=\mathbb{P}(X_{t+1}=y|X_t=x)=\frac{w(x,y)}{\sum_{z\in \mathcal{V}}w(x,z)}
$$
当$t\to \infty$，有$\forall x,p^t(x,\cdot)$会收敛到唯一稳定的分布，记作$\pi\in R^N_+$，正比于节点的度$d\in R^N_+$，即$\pi \propto d$

### Dirichlet energy

定义图函数$f$的迪利克雷能量
$$
D(f)=\sum_{x,y\in \mathcal{V}}\pi(x)p(x,y)|f(x)-f(y)|^2
$$
定义有向图下的随机游走拉普拉斯矩阵$L_{RW}$和未归一化的拉普拉斯矩阵$L$
$$
L_{RW}=I-\frac{1}{2}(P+\Pi^{-1}P^T\Pi)\\
L=\Pi-\frac{1}{2}(\Pi P+P^T \Pi)
$$
则$D(f)$也可表示成
$$
\begin{align}
D(f)&=\sum_{x,y\in \mathcal{V}}\pi(x)p(x,y)|f(x)-f(y)|^2\\
&=\sum_{x,y}\pi(x)p(x,y)f^2(x)+\sum_{x,y}\pi(x)p(x,y)f^2(y)-2\sum_{x,y}\pi(x)p(x,y)f(x)f(y)\\
&=\sum_x\pi(x)f^2(x)\sum_y p(x,y) + \sum_yf^2(y)\sum_x\pi(x)p(x,y) - 2\sum_xf(x)\sum_y\pi(x)p(x,y)f(y)\\
&=\left<f,\Pi f\right> + \left<f,\Pi f\right> - 2\left<f,\Pi P f\right>\\
&=\left<f,(2\Pi+\Pi P + P^T\Pi) f\right>\\
&=2\left<f,Lf\right>\\
&=2\left<f,L_{RW}f\right>_{\pi}
\end{align}
$$

## Generalized Dirichet energy

定义有向图上边的度量操作为$Q\{q(x,y)\}_{x,y\in V}$，则图函数$f$的广义上的迪利克雷能量(GDE)为
$$
D^2_Q(f)=\sum_{x,y}q(x,y)|f(x) - f(y)|^2
$$
当$q(x,y)=\pi(x)p(x,y)$时，$(4)$的$D(f)$是$D^2_Q(f)$的特例

$q(x,y)$可以结合任意的点度量函数和基于转移矩阵$P$的边度量函数，因此可重写上式为
$$
D^2_{v,P}(f)=\sum_{x,y}v(x)p(x,y)|f(x) - f(y)|^2
$$
定义点的正度量
$$
\xi(y) = \sum_{x\in \mathcal{V}}v(x)p(x,y)
$$
令$N=diag(v), \Xi = diag(\xi)$，定义广义的随机游走拉普拉斯矩阵$L_{RW}$和未归一化的拉普拉斯矩阵$L$
$$
L_{RW}(v)=I-(I+N^{-1}\Xi)^{-1}(P+N^{-1}P^TN)\\
L(v)=N+\Xi-(NP+P^TN)
$$
则$(14)$也可以表达为
$$
\begin{align}
D^2_{v,P}(f)&=\sum_{x,y}v(x)p(x,y)|f(x) - f(y)|^2\\
&=\sum_{x,y}v(x)p(x,y)f^2(x)+\sum_{x,y}v(x)p(x,y)f^2(y)-2\sum_{x,y}v(x)p(x,y)f(x)f(y)\\
&=\left<f,Nf\right>+\left<f,\Xi f\right>-2\left<f,NPf\right>\\
&=\left<f,(N+\Xi - (NP+P^TN))f\right>\\
&=\left<f,(N+\Xi)^{-1}(N+\Xi - (NP+P^TN))f\right>_{v+\xi}\\
&=\left<f,I - (N+\Xi)^{-1}N(P+N^{-1}P^TN)f\right>_{v+\xi}\\
&=\left<f,I - (I+N^{-1}\Xi)^{-1}(P+N^{-1}P^TN)f\right>_{v+\xi}\\
&=\left<f,L_{RW}(v)f\right>_{v+\xi}
\end{align}
$$
最后，我们引入标准化的GDE
$$
\bar{D}^2_{v,P}(f) = \frac{D^2_{v,P}(f)}{||f||^2_{v+\xi}}
$$
给定一个正的点度量$\mu$，可以引入一个参数化的点度量$v_t$
$$
v_t(x)=\mu^TP^t\delta_x
$$
其中$\delta_x$
