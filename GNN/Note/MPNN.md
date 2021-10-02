### Message Passing Neural Networks(MPNN)

**消息传播神经网络**是一种通用框架

节点的表示向量通过消息函数$M$和更新函数$U$进行$K$轮消息传播的迭代得到
$$
m_i^{(k+1)}=\sum_{v_j \in N(v_i)}M^{(k)}(h_i^{(k)},h_j^{(k)},e_{ij})
$$

$$
h_i^{(k+1)}=U^{(k)}(h_i^{(k)},m_i^{(k+1)})
$$

* $e_{ij}$表示边上的特征向量



##### 在GCN模型中

$$
M(h_{i}^{(k)},h_j^{(k)})=\hat{L}_{sym}[i,j]W^{(k)}h_j^{(k)}
$$


$$
U(m_i^{(k+1)})=\sigma(m_i^{(k+1)})
$$


##### 在R-GCN模型中

$$
M(h_j^{(k)},r)=\frac{1}{c_{i,r}}W_r^{(k)}h_j^{(k)}
$$


$$
U(h_i^{(k)},m_i^{(k+1)})=\sigma(m_i^{(k+1)}+W_oh_i^{(k)})
$$


##### 在GraphSAGE模型中

$$
\sum_{v_j \in N(v_i)} M(h_j^{(k)})=Agg\{h_j^{(k)}|v_j \in N(v_i)\}
$$

* 其中，$N(v)$为邻居采样函数

$$
U(h_i^{(k)},m_i^{(k+1)})=\sigma(W^{(k)}[m_i^{(k+1)}||h_i^{k}])
$$

