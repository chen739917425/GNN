# Fast Sequence Based Embedding with Diffusion Graphs

## 背景

随机游走方法向外扩散得相对较慢，且可能重复遍历一个节点多次，从而生成冗余的信息

## 本文工作

### 特征提取

给定图$G(V,E,X)$

先通过随机游走得到若干条节点序列，设置窗口半径为$\hat{w}$

对于每个节点$v$，我们计算其对应的命中频率向量$y_{v}\in R^{2\cdot\hat{w}\cdot|V|}=y_{v,+}||y_{v,-}$

$y_{v,+k}\in R^{|V|}$中第$i$个分量表示，在所有随机游走序列中，节点$v$的后继第$k$个节点位置上，节点$i$的出现次数之和

$y_{v,-\hat{k}}$同理，考虑的是前驱节点

### 从特征中学习embedding

由上我们得到了$R^{|V|\times 2\cdot\hat{w}\cdot|V|}$的特征，我们希望将其降维到$R^{|V|\times d}$

### 序列生成算法

