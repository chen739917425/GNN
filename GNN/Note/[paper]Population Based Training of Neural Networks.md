# Population Based Training of Neural Networks  

## 背景

神经网络在许多机器学习领域都表现出色，尤其是在强化学习和监督学习中。

但一个特定的神经网络模型的表现好坏，往往取决于模型的结构，数据的表示，优化的细节等等因素的联合调整，其中每个环节都由一些超参数控制

超参数的调整会影响学习的过程，也关系着模型的表现

通常这些超参需要凭经验预先设计，需要耗费大量的精力和时间去调试，有时并不能取得好的效果

特别是在强化学习中，理想的超参数往往是极度不固定的，需要随着环境的变化而调整，而不能事先确定

常见的两种调参方式有**并行搜索**和**串行优化**

* 并行搜索是指并行地运行多个模型优化程序（即神经网络的训练），这些程序使用不同的超参数训练神经网络，从这些程序中选择一个效果最好的
* 串行优化是指串行地多次运行模型优化程序，每次调整超参数可以使用之前运行得到的信息，常见的例子就是手工调参。这种方法最终往往能获得较好的解，但面对训练耗时比较大的模型，这种方式就不太可行

## 本文工作

### 任务

机器学习的常见形式是优化模型$f$的参数$\theta$，来最大化给定的目标函数$\hat{\mathcal{Q}}$（如分类，重构，预测等任务）

但一般来说，我们关心的模型真实性能$\mathcal{Q}$（例如模型在验证集上的准确率等）与目标函数$\hat{\mathcal{Q}}$（例如模型输出的预测标签与真实标签的交叉熵损失）是有差别的

我们定义函数$eval$来计算$Q$，并忽略模型除$\theta$以外的细节，则模型的优化任务如下
$$
\theta^*=argmax_{\theta\in\Theta}eval(\theta)
$$
即寻找模型参数$\theta^*$来使得模型性能$Q$最大

若采用迭代的方式优化模型，在给定超参$h$的情况下，参数$\theta$在每次迭代步骤后被更新，即
$$
\theta\gets step(\theta|h)
$$
假设整个优化过程有$T$次迭代，$\theta$的寻优可以如下表示
$$
\theta^*=optimise(\theta|\mathbf{h})=optimise(\theta|(h_t)_{t=1}^T)=step(step(\cdots step(step(\theta|h_1)|h_2)\cdots)|h_T)
$$

其中$\mathbf{h}=\{h_1,h_2,\cdots,h_T\}$是一个关于迭代次数的参数序列

这样迭代式的优化是计算昂贵的，取决于迭代的步数和单步迭代的计算代价。而且，最终的优化结果对$\mathbf{h}$的选择十分敏感，不正确地选择会导致收敛到一个坏的结果甚至无法收敛。要想正确地选择$\mathbf{h}$需要强大的先验（一般通过尝试不同的$\mathbf{h}$进行多次优化程序，如人工调参）

由于每个优化步骤都有对应的超参，那么超参数的可能取值是指数级别的。因此，一般的做法是设置$h_t$都相同（比如恒定的学习率）或者设定一个简单的自适应调整（比如学习率衰减）

### Population Based Training(PBT)

本文希望使用快速且计算高效的方式执行如下工作
$$
\theta^*=optimise(\theta|h^*),h^*=\mathrm{argmax}_{\mathbf{h}\in\mathcal{H}^T}eval(optimse(\theta|\mathbf{h}))
$$
由此，作者提出了**Population Based Training(PBT)**，其主要目的是提供一种方法，来协同优化待训练参数$\theta$和超参数$h$，从而获得更好的$\mathcal{Q}$

**PBT**将$N$个模型$\{\theta^i\}_{i=1}^N$看作一个种群$\mathcal{P}$，其中每个模型的初始超参$\mathbf{h}^i$设定不同

其算法流程如下

* 对于种群$\mathcal{P}$中的每一个个体$(\theta,h,p,t)$	//其中每个个体是异步并行的；$p$是模型的性能评估值，即$Q$

  * 若该个体还未收敛

    * 进行一步优化迭代，$\theta\gets step(\theta|h)$

    * 计算当前的模型性能，$p\gets eval(\theta)$

    * 若个体距上一次参数更新已经历了足够的迭代次数，即$ready(p,t,\mathcal{P})$为真

      * 获取种群其余个体中性能较好的参数，$h',\theta'\gets exploit(h,\theta,p,\mathcal{P})$

      * 若$\theta\ne \theta'$
        * 对$h'$进行微扰或重新采样，替换个体原来的$h$，用$\theta'$替换$\theta$，即$h,\theta\gets explore(h',\theta',\mathcal{P})$
        * 计算当前的模型性能，$p\gets eval(\theta)$

    * 更新当前个体为$(\theta,h,p,t+1)$

* 最终返回种群$\mathcal{P}$中$p$最优的个体

算法中

* $step()$表示模型在优化过程（如梯度下降）中的进行一次迭代
* $eval()$表示进行一次模型的性能评估（如在验证集上进行一次ACC指标测试）

* $ready()$表示判断当前模型个体是否准备就绪，可以进行参数更新。一般判断的标准为距上一次参数更新是否已迭代了足够多步

* $exploit()$判断当前个体的工作是否应该被停止并舍弃，然后用更优的个体替换它
* $explore()$对模型个体的超参数进行扰动或重采样，从而获得新的超参数，是算法具有一定的搜索寻优能力

$exploit,explore$是算法的关键，其实现方式取决于具体的应用场景

## 实验

### Reinforcement Learning

* **Hyperparameters**：UNREAL中的超参为学习率，entropy cost和unroll length；FuN中的超参为学习率，entropy cost和intrinsic reward cost 

* **Step**：RMSProp优化算法中的一步迭代

* **Eval**：last 10 episodic rewards  

* **Ready**：距上次参数更新已经过了$1\times10^6$到$10\times10^6$步

* **Exploit**：考虑两种策略
  * 从种群的其余个体中等概率抽取一个，若其性能评估$eval$优于当前个体，则当前个体的权重参数与超参数被其取代
  * 将种群中所有个体按性能评估排序，若当前个体位于末尾的$20\%$，则从靠前的$20\%$中随机抽取一个，取代当前个体的权重参数与超参数

* **Explore**：考虑两种策略

  * 微扰，每个超参数独立随机地被扰动，扰动的因子为$1.2$或$0.8$

  * 重采样，每个超参数从它原有的分布中重新采样获得

## Machine Translation

* **Hyperparameters**：学习率，attention dropout，layer dropout和ReLU dropout rates

* **Step**：Adam优化算法中的一步迭代

* **Eval**：BLEU score

* **Ready**：每$2\times10^3$步更新一次

* **Exploit**：从种群的其余个体中等概率抽取一个，若其性能评估$eval$优于当前个体，则当前个体的权重参数与超参数被其取代

* **Explore**：微扰，每个超参数独立随机地被扰动，扰动的因子为$1.2$或$0.8$

  

## Generative Adversarial Networks

* **Hyperparameters**：判别器学习率和生成器学习率

* **Step**：使用Adam优化算法，判别器进行五步迭代，然后生成器进行一步迭代

* **Eval**：Inception score的变体

* **Ready**：每$5\times10^3$步更新一次

* **Exploit**：与Reinforcement Learning任务中的设定类似

* **Explore**：微扰，每个超参数独立随机地被扰动，扰动的因子为$2.0$或$0.5$

