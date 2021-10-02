## 图卷积神经网络

### 图卷积

给定两个图信号$f_1,f_2$，定义卷积运算如下
$$
\begin{align}
f_1*f_2&=IGFT(GFT(f_1) \odot GFT(f_2))\\
&=V((V^Tf_1)\odot(V^Tf_2))\\
&=V(\hat{f_1}\odot(V^Tf_2))\\
&=V(diag(\hat{f}_1)(V^Tf_2))\\
&=(Vdiag(\hat{f}_1)V^T)f_2\\
&=H_{\hat{f}_1}f_2
\end{align}
$$

* 其中$\odot$表示哈达玛积，即对应元素相乘的运算

* 图卷积总能化为相应的图滤波运算，因此图卷积等价于图滤波

### 图卷积神经网络

#### 参数化频率响应矩阵

$$
\begin{align}
X'&=\sigma(V\begin{pmatrix} \theta_1&&& \\ &\theta_2&& \\ &&\ddots& \\ &&&\theta_N \end{pmatrix}V^TX)\\
&=\sigma(Vdiag(\theta)V^TX)\\
&=\sigma(\Theta X)
\end{align}
$$

$\sigma(\cdot)$是激活函数，$\theta=\begin{pmatrix}\theta_1\\\theta_2\\\vdots\\\theta_N\end{pmatrix}$是需要学习的参数，$X$是输入的图信号矩阵，$X'$是输出的图信号矩阵，$\Theta$是图滤波器

* 参数数量与节点数一致，在大规模图上极容易发生过拟合

#### 参数化多项式系数

$$
\begin{align}
X'&=\sigma(V(\sum_{k=0}^{K}\theta_k\Lambda^k)V^TX)\\
&=\sigma(Vdiag(\Psi\theta)V^TX)
\end{align}
$$

其中
$$
\Psi=\begin{pmatrix}
1&\lambda_1&\cdots&\lambda_1^K\\
1&\lambda_2&\cdots&\lambda_2^K\\
\vdots&\vdots&\ddots&\vdots\\
1&\lambda_N&\cdots&\lambda_N^K
\end{pmatrix}
$$

* K可以自由控制，一般设$K\ll N$，降低模型过拟合的风险

####  固定的图滤波器

$$
X'=\sigma(\tilde{L}_{sym}XW)\\
\tilde{L}_{sym}=\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}\\
\tilde{A}=A+I\\
\tilde{D}=D+I
$$

* $\tilde{L}_{sym}$的特征值范围为$(-1,1]$
* 上式称为**图卷积层**，以此为主体，堆叠多层的神经网络模型称为**图卷积模型$(GCN)$**

### GCN的低通滤波性质

$$
\begin{align}
\tilde{L}_{sym}&=\tilde{D}^{-\frac{1}{2}}(\tilde{D}-L)\tilde{D}^{-\frac{1}{2}}\\
&=I-\tilde{D}^{-\frac{1}{2}}L\tilde{D}^{-\frac{1}{2}}\\
&=I-\tilde{L}_s
\end{align}
$$

显然$\tilde{L}_s$仍然是实对称矩阵，可以正交对角化

令$\tilde{L}_s=V\tilde{\Lambda}V^T$，$\lambda_i$是$\tilde{L}_s$的特征值，可以证明$\lambda_i \in [0,2)$
$$
\tilde{L}_{sym}=I-V(\tilde{\Lambda})V^T=V(\mathrm{1}-\tilde{\Lambda})V^T
$$
则该滤波器的频率响应函数$h(\lambda)=1-\tilde{\lambda}_i\in(-1,1]$，这是线性收缩函数，起低通滤波作用

### GCN的过平滑

使用卷积层堆叠多层的GCN进行学习后，每个节点的向量表示会趋同，失去区分度，称为过平滑（Over-smooth）问题

#### 缓解方法

##### 空域角度

每层邻居聚合后的结果都会通过跳跃连接直接送往最终的聚合层，聚合层采用拼接、池化等操作进行聚合，作为最终的输出

##### 频域角度

调节滤波器的值，如增加$\tilde{A}$中节点自连接的权重
$$
A^{'}_{ij}=\begin{cases}\frac{p}{deg(v_i)}A_{ij},&if&i \ne j\\1-p,&if&i=j\end{cases}
$$
$p$趋于$0$时，模型趋于不聚合邻居的信息；$p$趋于$1$时，模型趋于不适用自身的信息
