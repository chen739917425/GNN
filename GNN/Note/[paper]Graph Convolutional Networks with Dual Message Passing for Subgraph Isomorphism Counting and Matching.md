# Graph Convolutional Networks with Dual Message Passing for Subgraph Isomorphism Counting and Matching

## 前置

### line graph

给定一个图$G$，点集为$V$，边集为$E$

令$G$的**line graph**为$G'=L(G)$，点集为$V'$，边集为$E'$

$V'$中的点$v'$是原图$G$中的边$e$，即$v'=g(e)$，其中函数$g:E_G\to V_{G'}$将$G$中的边转换为$G'$中的点

$G'$中有点$v’_i=g(e_i)$到点$v'_j=g(e_j)$的边**当且仅当**原图$G$中存在一个点$v$是$e_i$的出点且是$e_j$的入点

## 同构

存在一个映射$f:V_{G_1}\to V_{G_2}$，使得
$$
X_{G_1}(v)=X_G{_2}(f(v))\\
X_{G_2}(v')=X_G{_1}(f^{-1}(v'))\\
Y_{G_1}(e_{u,v})=Y_{G_2}(e_{f(u),f(v)})\\
Y_{G_2}(e_{u',v'})=Y_{G_1}(e_{f^{-1}(u'),f^{-1}(v')})\\
$$
则$G_1$与$G_2$同构，记作$G_1\simeq G_2$

可证明，$G_1\simeq G_2 \to L(G_1)\simeq L(G_2)$

##### 子图同构

若$G_{1_S}$是$G_1$的子图且$G_{1_S}\simeq G_2$，$G_1$对于$G_2$含有子图同构

### Whitney isomorphism theorem

对于节点数超过$4$的连通简单图，原图的同构映射$f$和其**line graph**的同构映射$f'$是一一对应的

即若$G_1\simeq G_2$且其同构映射为$f$，则$f$唯一对应$H_1=L(G_1)$与$H_2=L(G_2)$的同构映射$f'$

#### 拓展

对于连通的有平行边的有向异构图，为每条边添加带有特殊标记的反向边，则这样的图同构映射与其**line graph**的同构映射是一一对应的

 这个推论也可以扩展到子图同构中

上述结论说明了对于子图同构，**line graph**也保留了原图的结构信息

## Dual Message Passing Neural Networks

### 原图上卷积

$$
\Theta h\approx (\theta_0-\theta_1)h+\frac{2\theta_1}{\lambda_{\mathcal{G}max}}L_{\mathcal{G}}h
$$

其中

* $h\in R^n$是$n$个节点的标量特征

* $L_{\mathcal{G}}$是图$\mathcal{G}$的拉普拉斯矩阵
* $\lambda_{\mathcal{G}max}$是$L_{\mathcal{G}}$最大的特征值，且上界为$\max\{d_u+d_v|(u,v)\in E_\mathcal{G}\}$

### line graph上卷积

对于输入的图$\mathcal{G}$，将其转化为$\mathcal{H}=L(\mathcal{G})$

在$\mathcal{H}$上的卷积操作为
$$
\Gamma * z\approx (\gamma_0-\gamma_1)z+\frac{2\gamma_1}{\lambda_{\mathcal{H}max}}L_{\mathcal{H}}z
$$
其中$z\in R^m$是$\mathcal{H}$上$m$个点的标量特征

$\mathcal{H}$的边集的大小$|E_{\mathcal{H}}|=\frac{1}{2}\sum_{v\in V_{\mathcal{G}}}d_v^2-n=O(m^2)$，因此上式的计算代价是$O(m^2)$的，其中$m=|E_{\mathcal{G}}|=|V_{\mathcal{H}}|$

### Dual Message Passing Mechanism

使用一种计算代价为$O(m)$的异步方式结合原图和**line graph**上的卷积

#### 原图上

对于一张$n$个点，$m$条边的有向图$\mathcal{G}$，定义矩阵$B_{\mathcal{G}}\in R^{n\times m}$如下：
$$
b_{ve}=
\begin{cases}
1 & if \ \ v是边e的出点\\
-1 & if \ \ v是边e的入点\\
0 & otherwise
\end{cases}
$$
则有
$$
A_{\mathcal{G}}+A_{\mathcal{G}}^T=D_{\mathcal{G}}^+ + D_{\mathcal{G}}^- - B_{\mathcal{G}}B_{\mathcal{G}}^T
$$
特别地，如果$\mathcal{G}$添加了反向边，则有
$$
B_{\mathcal{G}}B_{\mathcal{G}}^T=2(D_{\mathcal{G}}-A_{\mathcal{G}})=2L_{\mathcal{G}}
$$
其中$L_{\mathcal{G}}$是拉普拉斯矩阵



#### line graph上

对于一张$n$个点，$m$条边的有向图$\mathcal{G}$，$\mathcal{H}$是其**line graph**

定义$\hat{B}_{\mathcal{G}}\in R^{n\times m}$如下：
$$
\hat{b}_{ve}=
\begin{cases}
1 & if \ \ v被边e关联\\
0 & otherwise
\end{cases}
$$
则有
$$
A_{\mathcal{H}}+A_{\mathcal{H}}^T=\hat{B}_{\mathcal{G}}^T\hat{B}_{\mathcal{G}}-2I_m
$$
其中$I_m\in R^{m\times m}$为单位矩阵

特别地，如果$\mathcal{G}$添加了反向边，则$\mathcal{H}$也含有反向边，则有
$$
A_{\mathcal{H}}=\frac{1}{2}\hat{B}_{\mathcal{G}}^T\hat{B}_{\mathcal{G}}-I_m
$$
进一步地
$$
L_{\mathcal{H}}=D_{\mathcal{H}}-A_{\mathcal{H}}=D_{\mathcal{H}}-\frac{1}{2}\hat{B}_{\mathcal{G}}^T\hat{B}_{\mathcal{G}}+I_m
$$

#### 结合

使用$(6)$改写$(2)$可得
$$
\Theta h\approx (\theta_0-\theta_1)h+\frac{\theta_1}{\lambda_{\mathcal{G}max}}B_{\mathcal{G}}B_{\mathcal{G}}^Th
$$
其中$B_{\mathcal{G}}^Th\in R^m$相当于在边空间中计算$\{x_v-x_u|(u,v)\in E_{\mathcal{G}}\}$

考虑用更好的滤波器来代替上述减法操作
$$
\Theta h\approx (\theta_0-\theta_1)h+\frac{\theta_1}{\lambda_{\mathcal{G}max}}B_{\mathcal{G}}z
$$
其中$z$是在边空间中执行某种特定计算的结果，涉及到$(3)$

使用$(10)$改写$(3)$可得
$$
\Gamma * z\approx (\gamma_0-\gamma_1)z+\frac{2\gamma_1}{\lambda_{\mathcal{H}max}}(D_{\mathcal{H}}+I_m)z-\frac{\gamma_1}{\lambda_{\mathcal{H}max}}\hat{B}_{\mathcal{G}}^T\hat{B}_{\mathcal{G}}z
$$
其中$\hat{B}_{\mathcal{G}}z\in R^{n}$对应$\{\sum_{(u,v)\in E_{\mathcal{G}}}z_{u,v}+\sum_{(u,v)\in E_{\mathcal{G}}}z_{v,u}|v\in V_{\mathcal{G}}\}$，使用$h$来替换可得
$$
\Gamma * z\approx (\gamma_0-\gamma_1)z+\frac{2\gamma_1}{\lambda_{\mathcal{H}max}}(D_{\mathcal{H}}+I_m)z-\frac{\gamma_1}{\lambda_{\mathcal{H}max}}\hat{B}_{\mathcal{G}}^Th
$$
其中

* $D_{\mathcal{H}}$在计算上不需要构造出**line graph**，仅与原图中的点的度数有关，即$\{d^-_{g(e)}=d^-_u,d^+_{g(e)}=d^+_v|(u,v)\in E_{\mathcal{G}}\}$

* 手工设置$\lambda_{\mathcal{H}max}=\max\{d_u+d_v|(u,v)\in E_{\mathcal{H}}\}=\max\{d_u^-+d_v^+|(u,v)\in E_{\mathcal{G}}\}$

最终，我们如下异步更新
$$
h^{(k)} = (\theta_0^{(k)}-\theta_1^{(k)})h^{(k-1)}+\frac{\theta_1^{(k)}}{\lambda_{\mathcal{G}max}}B_{\mathcal{G}}z^{(k-1)}
$$

$$
z^{(k)}=(\gamma_0^{(k)}-\gamma_1^{(k)})z^{(k-1)}+\frac{2\gamma_1^{(k)}}{\lambda_{\mathcal{H}max}}(D_{\mathcal{H}}+I_m)z^{(k-1)}-\frac{\gamma_1^{(k)}}{\lambda_{\mathcal{H}max}}\hat{B}_{\mathcal{G}}^Th^{(k-1)}
$$



#### 拓展

$(16)$中的$\hat{B}_{\mathcal{G}}$没有考虑边的方向，因此可将$(15),(16)$做如下拓展，考虑边的方向以及点、边的复合特征
$$
H^{(k)} = H^{(k-1)}W^{(k)}_{\theta_0}-(\hat{B}_{\mathcal{G}}-B_{\mathcal{G}})Z^{(k-1)}W^{(k)}_{\theta_1^-}+(\hat{B}_{\mathcal{G}}+B_{\mathcal{G}})Z^{(k-1)}W^{(k)}_{\theta_1^+}
$$

$$
Z^{(k)} = Z^{(k-1)}W^{(k)}_{\gamma_0}+2(D_{\mathcal{H}}+I_m)Z^{(k-1)}(W^{(k)}_{\gamma_1^-}-W^{(k)}_{\gamma_1^+})\\-(\hat{B}_{\mathcal{G}}-B_{\mathcal{G}})^TH^{(k-1)}W^{(k)}_{\gamma_1^-}\\+(\hat{B}_{\mathcal{G}}+B_{\mathcal{G}})^TH^{(k-1)}W^{(k)}_{\gamma_1^+}
$$

其中

* $H^k\in R^{n\times l^{(k)}},Z^k \in R^{m\times l^{(k)}}$分别是模型第$k$层的点和边的embedding
* $\hat{B}_{\mathcal{G}}-B_{\mathcal{G}}$滤掉了出边，$\hat{B}_{\mathcal{G}}+B_{\mathcal{G}}$滤掉了入边

