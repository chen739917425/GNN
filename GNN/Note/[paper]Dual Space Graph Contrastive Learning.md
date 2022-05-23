# Dual Space Graph Contrastive Learning

## 前置

### 双曲空间（Hyperbolic Space）

#### 曲率

曲率是描述几何体弯曲程度的量

直线的曲率为$0$

圆的曲率为常数，圆的半径越大，曲率越小

曲率有正负，可以如下理解：

* 正曲率：曲面上的三角形内角和大于$\pi$
* 负曲率：曲面上的三角形内角和小于$\pi$

**双曲空间是曲率为负常数的一类空间**，双曲空间随着半径增大呈现指数扩展，它可以看成是**连续的树结构空间**

#### Poincaré ball model

曲率为$-c$的庞加莱球模型的定义域为
$$
\mathbb{D}=\{(x_1,x_2,\cdots,x_n):x_1^2+x_2^2+\cdots+x_n^2<\frac{1}{c}\}
$$
该空间中两点$u,v$间的距离如下度量
$$
d_{\mathbb{D}}(u,v)=\frac{1}{arcosh(1+\frac{2||u-v||^2}{(1-||u||^2)(1-||v||^2)})}
$$
其中

* 曲率取常数$-1$

* **arcosh**是反双曲余弦函数

### 空间映射（Space Mapping）

空间映射有两种：指数映射和对数映射

指数映射将正切空间$\mathcal{T}_o\mathbb{D}_c$（原点为$o$的欧氏空间）映射为曲率为$-c$的双曲空间$\mathbb{D}_c$，对于$t\in\mathcal{T}_o\mathbb{D}_c$
$$
\exp_o^c(t)=tanh(\sqrt{c}||t||)\frac{t}{\sqrt{c}||t||}
$$
对数映射将$\mathbb{D}_c$映射为$\mathcal{T}_o\mathbb{D}_c$，对于$u\in\mathbb{D}_c$
$$
\log_o^c(u)=artanh(\sqrt{c}||u||)\frac{u}{\sqrt{c}||u||}
$$
神经网络中常见的三种方法有矩阵乘法，偏置加法和激活操作
$$
y=\sigma(W\cdot u+b)
$$
权重矩阵和偏置通常定义在欧氏空间下，因此不能直接对双曲空间下的embedding使用。因此我们要先将双曲空间下的embedding映射到欧式空间下进行运算，再映射回双曲空间

双曲空间下的矩阵乘法为
$$
W\otimes u=\exp^c_o(W\cdot \log_o^c(u))
$$
双曲空间下的偏置加法为
$$
u\oplus b=\exp_o^c(\log_o^c(u)+b)
$$
双曲空间下激活操作为
$$
y=exp_o^c(\sigma(\log_o^c(W\otimes u \oplus b)))
$$
通过上述转换后的方法，就可以将欧式空间中的图学习方法应用在双曲空间中

## 本文工作

### 子图采样

给定图$\mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{X})$，采样器为$S(\cdot)$
$$
SG=S(\mathcal{G})
$$
在欧式空间中使用**Diffusion Sample**方法

在双曲空间中使用**Community Structure Expansion Sampler**方法
