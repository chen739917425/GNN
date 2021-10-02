### Graph Attention Networks(GAT)

#### 注意力机制

##### 注意力系数

对于中心节点$v_i$，其邻居节点$v_j$与它的权重系数为
$$
e_{ij}=LeakyReLU(a^T[Wh_i||Wh_j])
$$

* 上述式子中，通过共享参数$W$对节点特征进行增维，然后拼接得到一个高维特征，最后通过一个单层的全连接层把它映射为一个实数

* $e_{ij}$表示的是$v_i,v_j$的相关度

使用$softmax$对权重系数做归一化，得到注意力系数
$$
\alpha_{ij}=\frac{exp(e_{ij})}{\sum_{v_k \in N(v_i)}exp(e_{ik})}
$$

##### 加权求和

加权求和得到节点$v_i$新的特征向量
$$
h_i'=\sigma(\sum_{v_j \in N(v_i)}\alpha_{ij}Wh_j)
$$

##### 多头注意力机制

调用$K$组相互独立的注意力机制
$$
h_i'=||_{k=1}^{K}\sigma(\sum_{v_j \in N(v_i)}\alpha_{ij}^{(k)}W^{(k)}h_j)
$$

* 其中$||$表示拼接操作，为了减少输出的维度，也可以将拼接操作改为平均操作

