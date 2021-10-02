## 图信号处理

### 傅里叶变换

$$
F(\omega)=\frac{1}{2\pi}\int_{-\infty}^{\infty}f(t)e^{-i\omega t}\mathrm{d}x
$$

* 函数$f(t)$可以拆解称无穷多个正弦波之和，$F(\omega)$表示角频率为$\omega$的正弦波的系数
* 傅里叶变换可以看作$f(t)$向基函数$e^{-i\omega t}$投影，$F(\omega)$则表示在$\omega$基上的坐标

* **基$e^{-i\omega t}$恰好是拉普拉斯算子的特征函数**
  $$
  \Delta e^{-i\omega t}=-\omega^2e^{-iwt}
  $$

### 图的傅里叶变换

图$(graph)$的傅里叶变换并不是傅里叶变换通过严谨的数学推导扩展到图$(graph)$上，因此更像是一种类比
$$
L=V \Lambda V^T\\
\hat{f}=V^Tf\\
f=V\hat{f}
$$

* 拉普拉斯矩阵$L$是实对称矩阵，$V$为$L$的特征向量构成的矩阵，$\Lambda$是$L$的特征值构成的对角阵

* 改写$TV(x)$的形式
  $$
  \begin{align}
  TV(x)&=f^TLf\\
  &=f^TV \Lambda V^Tf\\
  &=\hat{f}^T\Lambda\hat{f}\\
  &=\sum_i^{N}\hat{f}_{\lambda_i}^2\lambda_i
  \end{align}
  $$

* 图的总变差是拉普拉斯矩阵的特征值的加权和，权重为傅里叶系数的平方

* 将特征值按大小有序排列在一起，可理解为对图信号平滑度的梯度刻画，因此可以类比传统傅里叶变换中的**频率**

* 特征值越低，频率越低，对应的傅里叶基就变化越缓慢，相近节点上的信号值趋于一致；反之，则频率越高，对应的傅里叶基变化剧烈，相近节点的信号非常不一致

* 将图信号$f$看作函数$f(i)$，自变量是节点编号

* $\hat{f}(\lambda)$对应频域函数，自变量是拉普拉斯矩阵的特征值，对应频率，因变量（即对应频率的傅里叶系数）对应幅值，代表图信号在该频率分量上的强度

* 因此，图傅里叶变换使用拉普拉斯矩阵的特征向量作为投影的基，就是在将一个图信号分解到不同平滑程度的图信号上，就像传统傅里叶变换将函数分解到不同频率的函数上一样

### 图的滤波器

对给定图信号的频谱中的各个频率分量进行增强或衰减
$$
\begin{cases}
g=Hf=\sum_{i=1}^{N}(h(\lambda_i)\hat{f}_{\lambda_i})v_i\\
H=V\Lambda_hV^T\\
\Lambda_h=\begin{pmatrix}h(\lambda_1)&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&h(\lambda_N)\end{pmatrix}
\end{cases}
$$

* $H$只在对角线以及有边存在的位置上有非零值
* $H$是作用在每个节点一阶子图上的变换
* 满足上述性质的矩阵称为**图位移算子**

* $$\begin{cases}H(x+y)=Hx+Hy\\H_1(H_2x)=H_2(H_1x)\end{cases}$$

* $\Lambda_h$为$H$的**频率响应矩阵**，$h(\lambda)$为$H$的**频率响应函数**

