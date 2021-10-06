## 核函数

将输入向量在特征空间（转换后的数据空间，可能是高维空间）中进行点积的函数称为**核函数**

核函数度量两个向量在特征空间中的**相似性**

令$\varphi(x)$是$x$的高维映射，定义核函数为
$$
K(x,y)=\varphi(x)^T\varphi(y)
$$

### Mercer定理

设$X=\{x_1,x_2,\cdots,x_n\}$是一个有限数据集，来自空间$\mathcal{X}$

空间$\mathcal{X}$的Gram矩阵$K(X;k)\in R^{n\times n}$定义为
$$
K_{ij}=k(x_i,x_j)
$$
如果对$\forall X\in \mathcal{X}$，都有$K$为半正定矩阵，则称函数$k(\cdot,\cdot)$是半正定的

而根据**Mercer定理**，任何半正定函数都可以作为核函数

需要注意，**Mercer定理**是核函数的充分条件，但不是必要条件

### 优劣

优势

* 不需要找到映射函数$\varphi(\cdot)$显式地将数据映射到高维空间，而是直接在原始空间中计算两个样本在高维空间中的点积结果
* 以极低的计算成本在高维空间中寻找线性可分关系

劣势

* 为问题选定合适的核函数较为困难

### 常用核函数

##### 高斯核函数

$$
K(x,y)=exp(-\frac{||x-y||^2}{2\sigma^2})=exp(-\gamma||x-y||^2)
$$

考虑$f(x)$的泰勒展开
$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n
$$
令$a=0$，则$e^x$的泰勒展开为
$$
e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}
$$
对高斯核函数进行展开
$$
\begin{align}
K(x,y)&=exp(-\gamma||x-y||^2)\\
&=exp(-\gamma(x^2+y^2-2xy))\\
&=exp(-\gamma x^2)exp(-\gamma y^2)exp(\gamma 2xy)\\
&=exp(-\gamma x^2)exp(-\gamma y^2)\sum_{n=0}^{\infty}\frac{(2\gamma xy)^n}{n!}\\
&=exp(-\gamma x^2)exp(-\gamma y^2)\sum_{n=0}^{\infty}\sqrt{\frac{(2\gamma)^n}{n!}}x^n\sqrt{\frac{(2\gamma)^n}{n!}}y^n\\
&=exp(-\gamma x^2)exp(-\gamma y^2)h_x^Th_y\\
&=(exp(-\gamma x^2)h_x)^T(exp(-\gamma y^2)h_y)
\end{align}
$$
其中$h_x=\begin{pmatrix}1\\\sqrt{\frac{2\gamma}{1!}}x\\\sqrt{\frac{(2\gamma)^2}{2!}}x^2\\\vdots\end{pmatrix}$

由上可知，高斯核函数将样本映射到无限维的空间中，映射函数为$\varphi(x)=e^{-\gamma x^2}h_x$

因此，高斯核函数实现了在原始低维空间中计算两样本在无限维空间中的点积
