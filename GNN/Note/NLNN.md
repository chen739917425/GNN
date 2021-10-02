### Non-Local Neural Network(NLNN)

NLNN是对**注意力机制**的一般化总结

**non-local**操作
$$
h_i'=\frac{1}{C(h)}\sum_{\forall j}f(h_i,h_j)g(h_j)
$$

* 其中，$f$计算输入间的相似性，$g$对输入进行变换，$C(h)$用于归一化

* $g$一般选取线性变换

$f$的一些常见选择

##### 点积

$$
f(h_i,h_j)=\theta(h_i)^T\phi(h_j)
$$

* $\theta,\phi$都是线性变换，为了简便计算，$C(h)$可取值$N$，即向量$h$的数量

##### 全连接层

$$
f(h_i,h_j)=\sigma(w^T[\theta(h_i)||\phi(h_j)])
$$

* $w$是将向量映射到标量的权重向量，$C(h)$同样可选取为$N$

##### 高斯函数

使用高斯函数的扩展形式
$$
f(h_i,h_j)=e^{\theta(h_i)^T\phi(h_j)}
$$

* $C(h)=\sum_{\forall j}f(h_i,h_j)$

* 当$f$中$e$的幂指数项改为全连接层的形式，该做法就是GAT中的注意力机制。因此GAT可以看作NLNN的一个特例
