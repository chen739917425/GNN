### 梯度

$$
\nabla=\begin{pmatrix}\frac{\partial}{\partial x}\\\frac{\partial}{\partial y}\\\frac{\partial}{\partial z}\end{pmatrix}
$$

### 散度

$$
\nabla \cdot f = \nabla^Tf = \frac{\partial f_x}{\partial x} + \frac{\partial f_y}{\partial y} + \frac{\partial f_z}{\partial z}
$$

* 其中$f$ 是一个矢量函数
* 表示当$x,y,z$有微小增加时，$f_x,f_y,f_z$的增量总和
* 具有旋转不变性

### 拉普拉斯算子


$$
\Delta=\nabla\cdot\nabla=\begin{pmatrix}\frac{\partial^2}{\partial x^2}\\\frac{\partial^2}{\partial y^2}\\\frac{\partial^2}{\partial z^2}\end{pmatrix}\\
\Delta f = \nabla \cdot \nabla f = \nabla^2 f = \frac{\partial^2 f_x}{\partial x^2} + \frac{\partial^2 f_y}{\partial y^2} + \frac{\partial^2 f_z}{\partial z^2}
$$

* 其中$f$为标量函数
* 拉普拉斯算子$(\Delta)$可以理解为梯度$(\nabla)$的散度$(\nabla\cdot)$
* 离散意义下

$$
\frac{\partial f}{\partial x} = f'(x) = f(x+1) - f(x)
$$

$$
\frac{\partial^2 f}{\partial x^2} = f''(x) = f'(x) - f'(x-1) = f(x+1)+f(x-1)-2f(x)
$$

* 离散的二维平面下

$$
\Delta f = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
$$

* 拉普拉斯算子计算了周围点与中心点的梯度差的总和
* 拉普拉斯算子得到的是对该点进行微小扰动后可能获得的总增益 （或者说是总变化）

### 拉普拉斯矩阵

* 推广到图$(graph)$上

$$
\Delta f_i = \sum_{j \in N}w_{ij}(f_i-f_j) = \sum_{j \in N}w_{ij}f_i-\sum_{j \in N}w_{ij}f_j = d_if_i - w_{i:}f
$$

* 其中$w_{ij}$表示节点$i,j$的边权，$i,j$相邻时$w_{ij}=1$，不相邻时$w_{ij} = 0$。$d_i$为节点$i$的度数，$w_{i:} = (w_{i1},w_{i2},\cdots,w_{iN})$,  $f = \begin{pmatrix}f_1\\f_2 \\ \vdots \\f_N \end{pmatrix}$
* 拓展到所有节点

$$
\Delta f = \begin{pmatrix} \Delta f_1 \\ \Delta f_2 \\ \vdots \\ \Delta f_N \end{pmatrix} = \begin{pmatrix} d_1 & \cdots & 0 \\ \vdots & \ddots & \vdots  \\  0 & \cdots & d_N \end{pmatrix}f - \begin{pmatrix} w_{11} & \cdots & w_{1N}\\ \vdots & \ddots & \vdots \\ w_{N1} & \cdots & w_{NN}\end{pmatrix}f = Df-Wf = (D-W)f = Lf
$$

* 其中$D$为节点度数的对角阵，$W$为邻接矩阵，$L=(D-W)$为拉普拉斯矩阵

#### 标准化拉普拉斯矩阵

$$
L_{sym}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}
$$

$$
L_{sym\ ij}=\begin{cases}
1&if\quad i=j\\
-\frac{1}{\sqrt{deg(v_i)deg(v_j)}}&if\quad e_{ij}\in E\\
0&otherwise
\end{cases}
$$



### 图的总变差

* 定义$TV(f)$为图的总变差


$$
\begin{align}
TV(f) &= f^TLf \\ 
&= \sum_{i \in N}f_i\sum_{j \in N}\omega_{ij}(f_i-f_j) \\
&=\sum_{i \in N}\sum_{j \in N}\omega_{ij}f_i^2-\omega_{ij}f_if_j\\
&=\frac{1}{2}(\sum_{i \in N}\sum_{j \in N}\omega_{ij}f_i^2+\sum_{i \in N}\sum_{j \in N}\omega_{ij}f_j^2)-\sum_{i \in N}\sum_{j \in N}\omega_{ij}f_if_j\\
&=\frac{1}{2}\sum_{i \in N}\sum_{j \in N}\omega_{ij}f_i^2+\omega_{ij}f_j^2-2\omega_{ij}f_if_j\\
&=\frac{1}{2}\sum_{i \in N}\sum_{j \in N}\omega_{ij}(f_i-f_j)^2\\
&= \sum_{e_{ij} \in E}(f_i-f_j)^2
\end{align}
$$

* 其中$e_{ij}$表示连接节点$i,j$的无向边
* 总变差对每条边信号的差值进行加和，刻画图整体的平滑程度

### 拉普拉斯矩阵的特征值与特征向量

设$L$的特征值从小到大分别是$\lambda_1,\lambda_2,\cdots,\lambda_n$，对应的特征向量是$v_1,v_2,\cdots,v_n$，$f$已经标准化，即$f^Tf=1$

* 由瑞利定理(Rayleigh Theorem)可知，$f=v_i$时，有$TV(x)=f^TLf=\lambda_i$

* 由于$TV(x)$是刻画图整体的平滑程度，因此$v_1$是最平滑的图信号，$v_n$是最不平滑的图信号

* 拉普拉斯矩阵的**特征值**的有序排列可以看作对图信号**平滑程度**的一种梯度刻画

