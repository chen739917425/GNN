# Graph Condensation for Graph Neural Networks

## 背景

随着大规模图任务的流行，图的点边规模往往达到百万级别，对图神经网络的训练和调参是巨大的挑战。因此很自然地想到去简化图来加速计算，以及便利图的存储，可视化和检索

目前主流的简化图的两种做法是**图稀疏化（graph sparsification）**和**图坍缩（graph coarsening）**

图稀疏化一般删去原图上的一些边来简化图，而图坍缩将原图上的子图视作超级节点从而减少点的数量

然而，如果图的节点上也有属性，稀疏化的简化效果则不理想，因为它不减少节点的属性。另一方面，这两种方法可能会保留不是最有利于下游任务的主特征属性

因此本文希望提出一种方法能够有效地简化图同时保留足够的信息用于训练GNN

## 本文工作

### 符号说明&任务

令图数据集为$\mathcal{T}=\{A,X,Y\}$

其中，$A\in R^{N\times N}$为邻接矩阵，$N$为节点数，$X\in R^{N\times d}$为节点特征矩阵，$d$为特征维数，$Y\in\{0,1,\cdots,C-1\}^N$为节点类别标签，$C$为类别数

**Graph Condensation**希望能学习到一个规模小的生成的图数据集$\mathcal{S}=\{A',X',Y'\}$，其中$A\in R^{N'\times N'},X\in R^{N'\times D},Y\in\{0,1,\cdots,C-1\}^{N'}$并且$N'\ll N$

任务目标为
$$
\min\limits_{\mathcal{S}}\mathcal{L}(GNN_{\theta_s}(A,X),Y)\\
s.t.\quad \theta_{\mathcal{S}}=\mathop{\arg\min}\limits_{\theta}\mathcal{L}(GNN_{\theta}(A',X'),Y')
$$
其中，$GNN_{\theta}$表示由$\theta$参数化的GNN模型，$\theta_{\mathcal{S}}$表示模型在数据集$\mathcal{S}$上训练得到的参数，$\mathcal{L}$是诸如交叉熵之类的损失函数

然后优化上式可能会使最终的结果对模型的初始化过拟合，为了能够泛化到模型随机初始化的分布$P_{\theta_{0}}$，考虑重写上式为
$$
\min\limits_{\mathcal{S}}E_{\theta_0\sim P_{\theta_0}}[\mathcal{L}(GNN_{\theta_s}(A,X),Y)]\\
s.t.\quad \theta_{\mathcal{S}}=\mathop{\arg\min}\limits_{\theta}\mathcal{L}(GNN_{\theta(\theta_0)}(A',X'),Y')
$$
其中$\theta(\theta_0)$表示$\theta$是作用于$\theta_0$的函数

### 梯度匹配（Gradient Matching)

为优化上述目标，一种选择是像数据蒸馏那样优化计算在$\mathcal{S}$上的损失$\mathcal{L}$，然后通过梯度下降来优化$\mathcal{S}$，但作者认为这样做的计算代价是昂贵的

因此作者选用了**梯度匹配（Gradient Matching)**的方法，通过让模型在原图和生成图上每一步训练得到的梯度匹配，使得两个模型的权重参数匹配，也就是希望在两张图上训练的模型最后能够收敛到一个相似的解

使两个模型的参数匹配的形式化描述如下
$$
\min_{\mathcal{S}}E_{\theta_0\sim P_{\theta_0}}[\sum_{t=0}^{T-1}D(\theta_t^{\mathcal{S}},\theta_t^{\mathcal{T}})]\\
\theta_{t+1}^{\mathcal{S}}=opt_{\theta}(\mathcal{L}(GNN_{\theta_t^{\mathcal{S}}}(A',X'),Y'))\\
\theta_{t+1}^{\mathcal{T}}=opt_{\theta}(\mathcal{L}(GNN_{\theta_t^{\mathcal{T}}}(A,X),Y))
$$
其中，$D(\cdot,\cdot)$是一个距离函数，$T$表示模型训练的迭代步数，$opt_\theta$是模型参数的更新规则，$\theta_{t+1}^{\mathcal{S}},\theta_{t+1}^{\mathcal{T}}$是模型在图$\mathcal{S},\mathcal{T}$上训练到第$t$步时的参数

对于$opt_\theta$考虑单步的梯度下降
$$
\theta_{t+1}^{\mathcal{S}}=\theta_{t}^{\mathcal{S}}-\eta\nabla_{\theta}\mathcal{L}(GNN_{\theta_t^{\mathcal{S}}}(A',X'),Y')\\
\theta_{t+1}^{\mathcal{T}}=\theta_{t}^{\mathcal{T}}-\eta\nabla_{\theta}\mathcal{L}(GNN_{\theta_t^{\mathcal{T}}}(A,X),Y)
$$
由于$D(\theta_{t}^{\mathcal{S}},\theta_{t}^{\mathcal{T}})$通常很小，上述目标可以简化为梯度匹配
$$
\min_{\mathcal{S}}E_{\theta_0\sim P_{\theta_0}}[\sum_{t=0}^{T-1}D(\nabla_{\theta}\mathcal{L}(GNN_{\theta_t}(A',X'),Y'),\nabla_{\theta}\mathcal{L}(GNN_{\theta_t}(A,X),Y))]
$$
其中$\theta_{t}^{\mathcal{S}},\theta_{t}^{\mathcal{T}}$用$\theta_t$替代

假设两张图上，在模型某一层中梯度分别为$G^{\mathcal{S}}\in R^{d_1\times d_2}$和$G^{\mathcal{T}}\in R^{d_1\times d_2}$，定义他们之间的距离$dis$如下
$$
dis(G^{\mathcal{S}},G^{\mathcal{T}})=\sum_{i=1}^{d_2}(1-\frac{G_i^{\mathcal{S}}\cdot G_i^{\mathcal{T}}}{||G_i^{\mathcal{S}}||\cdot||G_i^{\mathcal{T}}||})
$$
其中，$G_i^{\mathcal{S}},G_i^{\mathcal{T}}$表示梯度矩阵的第$i$列的向量

将距离函数$D$定义为每一层的$dis$之和

由于同时学习$A',X',Y'$的难度较大，因此将$Y'$固定住，保持其类分布与原标签$Y$的相同

#### 图上采样
