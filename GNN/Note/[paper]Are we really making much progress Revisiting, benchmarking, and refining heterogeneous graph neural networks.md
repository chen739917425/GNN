# Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks  

## 背景

目前（文章发表于2021 KDD) 提出的解决异构图问题的HGNNs模型，它们的数据处理和实验设定各异，以至于难以评估目前在异构图问题上到底已经取得了多少成就

## 本文工作

* 本文通过统一数据集划分，特征处理和性能评估的方法，对现有HGNNs模型进行测试
* 由于不适当的设定和数据泄露问题，此前的一些模型的性能被错误地汇报，现有的SOTA的HGNNs模型没有论文所提到的那样优异的性能
* GCN和GAT被严重低估，原始的GAT在合适的输入下能有超过现有HGNNs的表现
* 在多数异构图数据上，元路径不是必须的
* HGNNs还有很大的改善空间

* 本文提出了Simple-HGN模型

