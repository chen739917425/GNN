### 瑞利商与瑞利定理

瑞利商(Rayleigh Quotient)为
$$
R(A,x)=\frac{x^HAx}{x^Hx}
$$
其中，$A$是厄米特矩阵（Hermitian Matrix），$x$是非零向量，$H$ 表示共轭转置

对于实数矩阵而言，共轭转置等价于转置
$$
R(A,x)=\frac{x^TAx}{x^Tx}
$$
#### 定理

$A$的特征向量是瑞利函数的驻点，驻点处的值为对应的特征值，由此
$$
\lambda_{min}\le R(A,x) \le \lambda_{max}
$$

#### 证明

给出实数情况下的证明

由
$$
R(A,cx)=\frac{cx^TAcx}{cx^Tcx}=\frac{x^HAx}{x^Hx}=R(A,x)
$$
可知，对x进行缩放，不会影响瑞利商的值

令$x^Tx=1$，则$R(A,x)=x^TAx$

在约束条件$x^Tx=1$下，求解$R(A,x)$的极值

使用拉格朗日乘子法
$$
L(x,\lambda)=x^TAx-\lambda(x^Tx-1)
$$
令$\nabla_x L=\mathbf{0}$，可得
$$
\nabla_x L=2Ax-2\lambda x=\mathbf{0}\\
Ax-\lambda x=\mathbf{0}
$$
因此，当$x$取$A$的特征向量$v_i$时，瑞利商取得极值$\lambda_i$
$$
R(A,v_i)=\frac{v_i^TAv_i}{v_i^Tv_i}=\frac{v_i^T\lambda_iv_i}{v_i^Tv_i}=\lambda_i
$$
