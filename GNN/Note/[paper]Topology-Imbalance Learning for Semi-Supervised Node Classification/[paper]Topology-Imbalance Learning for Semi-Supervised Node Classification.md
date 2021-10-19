## Topology-Imbalance Learning for Semi-Supervised Node Classification



### 符号说明&任务



定义无向无权图
$$
\mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{L})
$$
$\mathcal{V}$为点集，由特征矩阵$X\in R^{n\times d}$表示，$n$为节点数（样本数），$d$为特征维度。

$\mathcal{E}$为边集，由邻接矩阵$A\in R^{n\times n}$表示

$\mathcal{L}$为带标签点集，$\mathcal{L}\subset\mathcal{V}$，通常$|\mathcal{L}|\ll|\mathcal{V}|$

$\mathcal{U}$为无标签点集，$\mathcal{U}=\mathcal{V}-\mathcal{L}$

训练集中节点划分为$k$类$\{C_1,C_2,\cdots,C_k\}$

$\delta$为标签率，$\delta=\frac{|\mathcal{L}|}{|\mathcal{V}|}$

在一张同质的联通图上，面对极度**“Topology-Imbalance”**的情况，预测$\mathcal{U}$的标签。半监督学习任务



### 概念



#### QINL(Quantity-Imbalance Node Representation Learning)

* 不同种类之间的带标签节点的数量不平衡，关注的是带标签节点的**数量**

#### TINL(Topology-Imbalance Node Representation Learning)

* 标签节点在图上的分布不平衡，关注的是带标签节点在图上**所处的拓扑位置**
* 标签节点的拓扑不平衡是普遍存在的问题（即使在数量上平衡）
* 可能导致存在无标签节点受不同标签的**影响冲突**，或者存在无标签节点受到的**影响过少**，供分类的信息不足
* 从研究对象上看，**TINL**独立于**QINL**，它关注决策边界上，每个带标签节点独特的拓扑位置带来的影响，因此**QINL**的应对策略不适用于**TINL**

![](image1.png)



### Topology-Imbalance



#### 通过LPA（标签传播算法）理解Topology-Imbalance

$$
f^{(k+1)}=\alpha f^{(0)}+(1-\alpha)A'f^{(k)}
$$

其中，$A'=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$，$f^{k}=\begin{pmatrix}f_1^k\\f_2^k\\\vdots\\f_n^k\end{pmatrix}$，$f_i^k$表示第$k$次迭代后节点$v_i$属于各个类的概率分布，$f^0$将带标签节点初始化为one-hot向量，无标签节点初始化为$0$向量

令$f=\alpha f^{(0)}+(1-\alpha)A'f$，可得$f$收敛于
$$
f=\alpha(I-(1-\alpha)A')^{-1}f^0
$$
对$f$级数展开
$$
\begin{align}
f&=\alpha\sum_{n=0}^{\infty}(1-\alpha)^n(A')^nf^0\\
&=\alpha(f^0+(1-\alpha)A'f^0+(1-\alpha)^2(A')^2f^0+\cdots)
\end{align}
$$
带标签节点对于其余节点的影响，随着拓扑距离的增大而衰弱



#### 通过influence conflict（影响冲突）来衡量Topology-Imbalance

##### Totoro（Conflict Detection Topology Relative Location metric)

定义一个**PageRank**矩阵$P$
$$
P=\alpha(I-(1-\alpha)A')^{-1}
$$
对于所有的**带标签节点**$v\in \mathcal{L}$，定义一个衡量influence conflict程度的Totoro指标
$$
T_v=\mathbb{E}_{x\sim P_v}[\sum_{j\in[1,k],j\ne y_v}\frac{1}{|C_j|}\sum_{i\in C_j}P_{i,x}]
$$
$T_v$越大，表示节点$v$受到的影响冲突越强烈，说明节点$v$越接近分类边界



#### ReNode

基于Totoro指标，设计每个标签节点$v\in \mathcal{L}$的训练权重
$$
w_v=w_{min}+\frac{1}{2}(w_{max}-w_{min})(1+cos(\frac{Rank(T_v)}{|\mathcal{L}|}\pi))
$$
其中，$w_{min},w_{max}$是超参数，表示权重的上下界；$Rank(T_v)$表示$T_v$在$\{T_i|i\in [1,l]\}$中的升序排名

由上，可得监督损失函数
$$
L_T=-\frac{1}{|\mathcal{L}|}\sum_{v\in\mathcal{L}}w_v\sum_c^ky_v^{*c}\log g_v^c
$$
其中，$g=softmax(\mathcal{F}(X,A,\theta))$

