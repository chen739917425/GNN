## Gaussian

### 高斯函数

$$
f(x)=ae^{-\frac{(x-b)^2}{2c^2}}
$$



* 一维高斯函数形如钟形曲线（bell curve），其中$a$为曲线峰顶的高度，$b$为曲线的对称中心，$c$与半峰全宽相关

#### 高斯分布

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

#### 二维高斯分布

$$
f(x,y)=\frac{1}{2\pi \sigma^2}e^{-(\frac{(x-\mu_x)^2}{2\sigma_x^2}+\frac{(y-\mu_y)^2}{2\sigma_y^2})}
$$

* 二维高斯分布可用于高斯滤波器，进行图像模糊。
* 设置滤波窗口中所有像素点的权重，使其权重符合二维高斯分布，然后进行加权平均。

* $\sigma$设置得越大，权重分布越均匀，图像越模糊，反之，权重越集中与中心点，图像趋于保持原有的清晰度

### 高斯核函数

$$
K(x_i,x_j)=e^{-\frac{||x_i-x_j||^2}{2\sigma^2}}=e^{-\gamma||x_i-x_j||^2}
$$

* 高斯核函数将样本映射到无限维的空间中计算点积

