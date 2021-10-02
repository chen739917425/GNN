# 线性代数的本质

教程地址：[线性代数的本质 - bilibili](https://www.bilibili.com/video/BV1ys411472E)

## 01 向量究竟是什么？

## 02 线性组合、张成的空间与基

### 什么是 “线性”

**定义**

- 可加性： $\large f(x+y)=f(x)+f(y)$
- 齐次性： $\large f(kx)=kf(x)$

即： $\large f(ax+by)=af(x)+bf(y)$



**物理意义**

可加性：因变量叠加后的作用结果 等于 各个因变量独自作用结果的叠加

齐次性：因变量缩放，因变量的作用结果也同等比例地缩放



**3Blue1Brown**

“如果固定其中一个标量，让另一个标量自由变化，所产生的向量的终点会描出一条直线。”

"If you fix one of those scalars and let the other one change its value freely, the tip of the resulting vector draws a straight line."

### 什么是 “线性组合”

两个数乘向量的和被称为着两个向量的**线性组合**。

So any time that you're scaling two vectors and adding them like this, it's called a **linear combination** of those two vectors.
$$
\large a\vec{\bold v}+b\vec{\bold w}
$$

### 什么是 “张成的空间”

所有可以表示为给定向量线性组合的向量的集合被称为给定向量**张成的空间 (span)** 。

The set of all possible vectors that you can reach with a linear combination of a given pair of vectors is called the **span** of those two vectors.

两个向量张成的空间实际上是问 —— 仅通过**向量加法**与**向量数乘**这两种基础运算，你能获得的所有可能向量的集合是什么。

The span of two vectors is basically a way of asking what are all the possible vectors you can reach using only these two fundamental operations - **vector addition** and **scalar multiplication**.

### 什么是 “线性相关”

- **线性相关 / Linearly dependent**

一组向量中至少有一个是多余的，没有对张成空间做出任何贡献。

At least one of these vectors is redundant, not adding anything to our span.

你有多个向量，并且可以移除其中一个而不减小张成的空间。

You have multiple vectors and you could remove one without reducing the span.

其中一个向量可以表示为其他向量的线性组合，因为它已经落在其他向量张成的空间之中。

One of the vectors can be expressed as a linear combination of the others, since it's already in the span of the others.

- **线性无关 / Linearly independent**

所有的向量都给张成的空间增添了新的维度。

Each vector really does add another dimension to the span.

## 03 矩阵与线性变换

### 什么是 “线性变换”

直观地说，如果一个变换具有以下两条性质，我们就称它是**线性**的：

Visually speaking, a transformation is linear if it has two properties:

- **直线依旧是直线** —— 直线在变换后仍然保持为直线，不能有所弯曲。

  **Lines remain lines** —— All lines must remain lines, without getting curved.

- **原点保持固定** —— 原点必须保持固定。

  **Origin remains fixed** —— The origin must remain fixed in place.

总的来说，你应该把线性变换看作是 “保持网格线**平行**且**等距分布**” 的变换。

In general, you should think of linear transformations as "keeping grid lines **parallel** and **evenly spaced**".



网格线保持平行且等距分布的性质有一个重要的推论：

The property that grid lines remain parallel and evenly spaced has a really important consequence:

向量 $\vec{\bold v}$ 是 $\hat{\bold \imath}$ 和  $\hat{\bold \jmath}$ 的一个特定线性组合，那么变换后的向量 $\vec{\bold v}$ 也是变换后 $\hat{\bold \imath}$ 和  $\hat{\bold \jmath}$ 的同样的线性组合。

It started off as a certain linear combination of i-hat and j-hat, and it ends up is that same linear combination of where those two vectors landed.
$$
\begin{align}
\large \vec{\bold v} &=\large -1\hat{\bold \imath} + 2\hat{\bold \jmath}\\
\large &=\large -1 \begin{bmatrix}1\\0\end{bmatrix} + 2 \begin{bmatrix}0\\1\end{bmatrix}\\\\
\large \rm Transformed\ \vec{\bold v} &=\large \rm -1(Transformed\ \hat{\bold \imath}) + 2(Transformed\ \hat{\bold \jmath})\\
\large &=\large -1 \begin{bmatrix}1\\-2\end{bmatrix} + 2 \begin{bmatrix}3\\0\end{bmatrix}
\end{align}
$$
更一般情况下的坐标 (x, y)

Write the vector with more general coordinates (x, y)
$$
\large \hat{\bold \imath} \rightarrow \begin{bmatrix}1\\-2\end{bmatrix} \qquad \hat{\bold \jmath} \rightarrow \begin{bmatrix}3\\0\end{bmatrix}\\ \ \\
\large \begin{bmatrix}x\\y\end{bmatrix} \rightarrow x\begin{bmatrix}1\\-2\end{bmatrix} + y\begin{bmatrix}3\\0\end{bmatrix} 
= \begin{bmatrix}1x+3y\\-2x+0y\end{bmatrix}
$$
最一般的情况下

Most general case
$$
\large \hat{\bold \imath} \rightarrow \begin{bmatrix}a\\c\end{bmatrix} \qquad \hat{\bold \jmath} \rightarrow \begin{bmatrix}b\\d\end{bmatrix}\\
用一个\ 2\times 2\ 矩阵表示线性变换\large \rightarrow \begin{bmatrix}a & b\\c & d\end{bmatrix}\\ \ \\
\large \begin{bmatrix}a & b\\c & d\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}
= \underbrace{ x\begin{bmatrix}a\\c\end{bmatrix} + y\begin{bmatrix}b\\d\end{bmatrix} }_{\rm Where\ all\ the\ intuition\ is\\ \quad\ \ \ 直观的部分在这里}
= \begin{bmatrix}ax+by\\cx+dy\end{bmatrix}
$$

## 04 矩阵乘法与线性变换复合

## 05 行列式

### 几何意义

我们关注以 $\hat\imath$ 为底边，以 $\hat\jmath$ 为左边的 $1\times 1$ 正方形。

We focus our attention on the 1-by-1 square, whose bottom sits on i-hat and whose left side sits on j-hat.

你只要知道这个**单位正方形面积变化的比例**，它就能告诉你其他**任意区域**的面积变化比例。

If you know how much the area of that one single unit square changes, it can tell you how the area of any possible region in space changes.

无论一个方格如何变化，对其他大小的方格来说，都会有相同的变化。

Whatever happens to one square in the grid has to happen to any other square in the grid, no matter the size.

这是由 “网格线保持平行且等距分布” 这一事实推断得出的。

This follows from the fact that grid lines remain parallel and evenly spaced.

对于不是方格的形状，它们可以用许多方格良好**近似**。

Any shape that's not a grid square can be **approximated** by grid squares pretty well.

### 简单定义

这个特殊的缩放比例，即线性变换改变面积的比例，被称为这个变换的**行列式**。

This very special scaling factor - the factor by which a linear transformation changes any area, is called the **determinant** of that transformation.

### 压缩 / 降维

一个二维线性变换的**行列式为 0** ，说明它将整个平面**压缩到一条线**，甚至是**一个点**上，此时任何区域的面积都变成了 0 。

The **determinant of a 2-D transformation is 0**, if it **squishes** all of space on to **a line**, or even onto **a single point**, since then, the area of any region would become 0.

### 负数的含义

当空间**定向**发生**改变**时，行列式为**负**，但行列式的**绝对值**依然表示区域面积的缩放比例。

Whenever the **orientation** of space is **inverted**, the determinant will be **negative**, the **absolute value** of the determinant still tells you the factor by which areas have been scaled.

### 计算公式的几何理解

![image-20210328200512358](E:\My Nutstore\Mathematics\Essense of Linear Algebra\Essense of Linear Algebra.assets\image-20210328200512358.png)

## 06 逆矩阵、列空间与零空间

### 什么是逆矩阵

线性方程组 / 向量方程

Linear system of equations / Vector equation
$$
\large A\vec{\bold x} = \vec{\bold v}
$$
矩阵 $A$ 代表一种线性变换，

The matrix $A$ corresponds with some linear transformation,

求解 $A\vec{\bold x} = \vec{\bold v}$ 意味着我们去寻找一个向量 $\vec{\bold x}$ ，使得它在变换后与 $\vec{\bold v}$ 重合。

Solving $A\vec{\bold x} = \vec{\bold v}$ means we're looking for a vector $\vec{\bold x}$ , which after applying the transformation lands on  $\vec{\bold v}$ .



#### 当 $\large \bf {\rm det}(A)\neq0$ 

当 $A$ 的行列式不为零时，有且仅有一个向量（在变换后）与 $\vec{\bold v}$ 重合。

When the determinant is non-zero, there will always be one and only one vector that lands on $\vec{\bold v}$ .

你可以通过逆向进行变换来找到这个向量。

You can find it by playing the transformation in reverse.
$$
\large A^{-1}A\vec{\bold x} = \vec{\bold x} = A^{-1}\vec{\bold v}
$$


#### 当 $\large \bf {\rm det}(A)= 0$ 

当 $A$ 的行列式为零时，与这个方程组相关的变换将空间压缩到更低的维度上，此时没有逆变换。

When the determinant is zero, and the transformation associated with this system of equations squishes space into a smaller dimension, there is no inverse.

你不能将一条线 “解压缩” 为一个平面，至少这不是一个函数能做的。

You cannot "unsquish" a line to turn it into a plane, at least, that's not something that a function can do.



即使不存在逆变换，解仍然可能存在。

It's still possible that a solution exists even when there is no inverse.

比如说，一个变换将空间压缩为一条直线，而向量 $\vec{\bold v}$ 恰好处于这条直线上。

It's just that when your transformation squishes space onto a line, and the vector $\vec{\bold v}$ lives somewhere on that line.

### 什么是秩

“**秩**” 代表着变换后空间的**维数**。

The word "**rank**" means **the number of dimensions** in the output of a transformation.

更精确的秩的定义是**列空间**的维数。

A more precise definition of rank would be that it's the number of dimensions in **the column space**.

当秩达到最大值时，意味着**秩与列数相等**，我们称之为 “**满秩**”。

When this rank is as high as it can be, meaning **it equals the number of columns**, we call the matrix "**full rank**". 

### 什么是列空间

所有可能的输出向量 $A\vec{\bold v}$ 构成的集合，被称为矩阵 $A$ 的 “列空间”。

Set of all possible outputs $A\vec{\bold v}$ is called the "column space" of matrix $A$ .



**为什么叫列空间**

矩阵的列告诉你基向量变换后的位置，这些变换后的基向量张成的空间就是所有可能的变换结果。

The columns of your matrix tell you where the basis vectors land, and the span of those transformed basis vectors gives you all possible outputs.

换句话说，列空间就是矩阵的**列**所**张成的空间**。

In other words, the column space is the **span** of the **columns** of your matrix.

### 什么是零空间

变换后落在原点（零向量）的向量的集合被称为矩阵的 “零空间” 或 “核”。

This set of vectors that lands on the origin (zero vector) is called the "null space" or the "kernel" of your matrix.

对线性方程组来说，当向量 $\vec{\bold v}$ 恰好为零向量时，零空间给出的就是这个向量方程所有可能的解。

In terms of the linear system of equations, when $\vec{\bold v}$ happens to be the zero vector, the null space gives you all of the possible solutions to the equation.
$$
\large A\vec{\bold x} = \begin{bmatrix}0\\0\end{bmatrix}
$$

## 附注 非方阵

### 低维到高维

一个 $3\times 2$ 矩阵，它的几何意义是将二维空间映射到三维空间上。

A 3-by-2 matrix, it has the geometric interpretation of mapping two dimensions to three dimensions.
$$
\large \begin{bmatrix}3 & 1\\4 & 1\\5 & 9\end{bmatrix}
$$
矩阵有**两列**表明输入空间有**两个基向量**，有**三行**表明每一个基向量在变换后都用**三个独立的坐标**来描述。

Since the **two columns** indicate that the input space has **two basis vectors**, and the **three rows** indicate that the landing spots for each of those basis vectors is described with **three separate coordinates**.
$$
\large \hat{\bold \imath} \rightarrow \begin{bmatrix}3\\4\\5\end{bmatrix} \qquad \hat{\bold \jmath} \rightarrow \begin{bmatrix}1\\1\\9\end{bmatrix}
$$

### 高维到低维

类似的，当你看到一个两行三列的 $2\times 3$ 矩阵。

Likewise, if you see a 2-by-3 matrix with two rows and three columns.
$$
\large \begin{bmatrix}3 & 1 & 4\\1 & 5 & 9\end{bmatrix}
$$
矩阵有**三列**表明原始空间有**三个基向量**，也就是说原始空间是**三维**的。

The **three columns** indicate that you're starting in a space that has **three basis vectors**, so we're starting in **three dimensions**.

有**两行**表明这三个基向量在变换后都**仅用两个坐标**来描述，所以它们一定落在**二维**空间中。

The **two rows** indicate that the landing spot for each of those three basis vectors is described with **only two coordinates**, so they must be landing in **two dimensions**.
$$
\large \hat{\bold \imath} \rightarrow \begin{bmatrix}3\\1\end{bmatrix} \qquad \hat{\bold \jmath} \rightarrow \begin{bmatrix}1\\5\end{bmatrix} \qquad \hat{k} \rightarrow \begin{bmatrix}4\\9\end{bmatrix}
$$
因此这是一个从**三维空间**到**二维空间**的变换。

So it's a transformation from **3-D space** onto the **2-D plane**.



**自我理解**

所谓线性变换，实际上是以当前基向量作为参照，得出线性变换后的基向量的样子。

- 如果是同维度的线性变换，相当于通过当前的基向量得到变换后基向量的坐标。
- 如果是高维度到低维度的线性变换，相当于只在高维空间指定低维空间所需要的坐标值，而其余维度的坐标值任意，这并不会影响到低维空间中基向量的样子（低维空间是高维空间的投影，投影之外的维度变化不会影响投影的样子）。所以映射后的低维空间在原高维空间中也可以任意形态展示，如一条过原点的直线（一维空间）在二维平面内任意旋转。线性变换只需要告诉低维空间基向量应有的长度就够了。
- 如果是低维度到高维度的线性变换，只是将低维空间放在一个高维空间中，其基向量张成的空间依旧是低维空间。

对于高维到低维空间的映射，注意到这里每个基向量实际上是省略了第三个维度（或更高维度）的坐标值，实际上这些坐标在高维空间中是可以存在，同时可以被指定为任意值，因为不管指定什么样的坐标值，都不会影响到高维空间中基向量在低维空间中的样子（低维空间是高维空间的投影，投影之外的维度变化不会影响投影的样子）。

## 07 点积与对偶性

### 与单位向量的点积

将一条数轴任意斜向放置在空间中，保持 $0$ 在原点位置，然后将**二维向量**直接**投影**到这条**数轴**上。

Place a number line diagonally on space somehow with the number $0$ sitting at the origin, we will **project 2-D vectors** straight onto this diagonal **number line**.

根据这个投影，定义一个从**二维向量**到**数**的**线性变换**，我们能找到描述这个变换的 $1\times 2$ **矩阵**。

With this projection, we just defined a **linear transformation** from **2-D vectors** to **numbers**, so we're going to be able to find some kind of **1-by-2 matrix** that describes that transformation. 

考虑变换后**基向量** $\hat{\bold \imath}$ 和 $\hat{\bold \jmath}$ 的位置，因为它们就是**矩阵的列**。

Think about where $\hat{\bold \imath}$ and $\hat{\bold \jmath}$ each land, since those landing spots are going to be **columns of the matrix**.

通过对称性，基向量落在数轴上的**投影长度**就等于该数轴的**单位向量** $\hat{u}$ 在二维空间中的**坐标值**。

By symmetry, the **number** where basis vectors lands when it's projected onto that diagonal number line is going to be the **coordinate value** of **unit vector** $\hat{u}$ of the number line.

所以描述投影变换的 $1\times 2$ 矩阵的两列就分别是 $\hat{u}$ **的两个坐标**。

So the entries of the 1-by-2 matrix describing the projection transformation are going to be the **coordinates of**  $\hat{u}$ .
$$
\large U = \begin{bmatrix}u_x & u_y\end{bmatrix}
$$
而空间中任意向量经过投影变换的结果，也就是**投影矩阵与这个向量相乘**。

And computing this projection transformation for arbitrary vectors in space, which requires **multiplying that matrix by those vectors**.
$$
\large q = U\ \vec{\bold q} = \begin{bmatrix}u_x & u_y\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=u_x\cdot x+u_y\cdot y
$$
这就是为什么与单位向量的点积可以解读为：

This is why taking a dot product with a unit vector can be interpreted as : 

将向量投影到单位向量所在的直线上所得到的投影长度。

Projecting a vector onto the span of that unit vector and taking the length.

### 与非单位向量的点积

我们把单位向量 $\hat{u}$ 放大为原来的 3 倍，数值上说，它的每个坐标都被放大为原来的 3 倍。

We scale $\hat{u}$ up by a factor of 3, numerically, each of its components gets multiplied by 3.

要寻找与这个向量相关的投影矩阵，实际上就是基向量投影得到的值的 3 倍。

Looking at the matrix associated with that vector, it takes basis vectors to $3$ times the values where they landed before.
$$
\large P = 3U = \begin{bmatrix}3u_x & 3u_y\end{bmatrix}
$$
新矩阵可以看作将任何向量朝斜着的数轴上投影，然后将结果乘以 3 （缩放）。

The new matrix can be interpreted as projecting any vector on the number line copy and multiplying where it lands by 3 (scale).
$$
\large q = P\ \vec{\bold q} = 3U\ \vec{\bold q} =  \begin{bmatrix}3u_x & 3u_y\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=3\begin{bmatrix}u_x & u_y\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=3(u_x\cdot x+u_y\cdot y)
$$
这就是为什么向量与给定非单位向量的点积可以解读为：

This is why the dot product with a non-unit vector can be interpreted as : 

首先朝给定向量上投影，然后将投影的值与给定向量长度相乘（先投影，后缩放）。

First projecting onto that vector, then scaling up the length of that projection by the length of the vector.

### 对偶性

设，存在向量 $\vec{\bold p}$ 与向量 $\vec{\bold q}$ ：
$$
\large \vec{\bold p}=\begin{bmatrix}a\\b\end{bmatrix}\qquad\qquad \vec{\bold q}= \begin{bmatrix}c\\d\end{bmatrix}
$$
向量 $\vec{\bold p}$ 与向量 $\vec{\bold q}$ 的点积为：
$$
\large \vec{\bold p}\cdot\vec{\bold q}=\begin{bmatrix}a\\b\end{bmatrix}\cdot \begin{bmatrix}c\\d\end{bmatrix} = ac+bd
$$
向量 $\vec{\bold p}$ 的对偶是由它定义的线性变换 $P$ ：
$$
\large P = \begin{bmatrix}a & b\end{bmatrix}
$$
设 $\large s = \sqrt{a^2+b^2}$ ，则 $P$ 所定义的线性变换表示将基向量 $\hat{\imath}$ 和 $\hat{\jmath}$ 分别变换为向量 $\vec{\bold p}$ 所在数轴上的 $\cfrac{a}{s}$ 和  $\cfrac{b}{s}$ ，再放缩 $\large s$ 倍。

即，任意向量 $\vec{\bold q}$ 在 $\vec{\bold p}$ 所在直线上的投影长度，乘以 $\large s$ 。
$$
\large P\ \vec{\bold q}=s\begin{bmatrix}\frac{a}{s} & \frac{b}{s}\end{bmatrix}\begin{bmatrix}c\\d\end{bmatrix} = s(\frac{a}{s}\ c+\frac{b}{s}\ d) = ac+bd
$$
直接地，可看作 $P$ 所定义的线性变换将基向量 $\hat{\imath}$ 和 $\hat{\jmath}$ 分别变换为向量 $\vec{\bold p}$ 所在数轴上的 $a$ 和  $b$ ，则任意向量 $\vec{\bold q}$ 在数轴上的投影长度为：
$$
\large P\ \vec{\bold q}=\begin{bmatrix}a & b\end{bmatrix}\begin{bmatrix}c\\d\end{bmatrix} = ac+bd
$$


**对偶性** $\Leftrightarrow$ 自然而又出乎意料的**对应关系**

**Duality** $\Leftrightarrow$ Natural-but-surprising **correspondence**

一个**向量**的对偶是由它定义的**线性变换**。

Dual of a **vector** is the **linear transformation** that it encodes.

一个多维空间到一维空间的**线性变换**的对偶是多维空间中**某个特定向量**。

Dual of a **linear transformation** from some space to one dimension is **a certain vector** in that space.

## 08 叉积

### 二维空间

对于二维向量的叉积， $\vec{\bold v}$ 叉乘 $\vec{\bold w}$ ，

For the 2-D cross product,  $\vec{\bold v}$ cross $\vec{\bold w}$ , 
$$
\large \vec{\bold v} = \begin{bmatrix}3 \\1\end{bmatrix} \qquad \vec{\bold w} = \begin{bmatrix}2\\-1\end{bmatrix}
$$
你需要将 $\vec{\bold v}$ 的坐标作为矩阵的第一列， $\vec{\bold w}$ 的坐标作为矩阵的第二列，然后直接计算行列式。

What you do is you rite the coordinates of  $\vec{\bold v}$ as the first column of the matrix, and you take the coordinates of $\vec{\bold w}$ and make them the second column, then you just compute the determinant.
$$
\large \vec{\bold v}\times \vec{\bold w} = {\rm det}(\begin{bmatrix}3 & 2\\1 & -1\end{bmatrix})
$$
这是因为，由 $\vec{\bold v}$ 和 $\vec{\bold w}$ 的坐标为列所构成的矩阵，与一个将 $\hat{\imath}$ 和 $\hat{\jmath}$ 分别移至 $\vec{\bold v}$ 和 $\vec{\bold w}$ 的**线性变换**相对应。

This is because a matrix whose columns represent $\vec{\bold v}$ and $\vec{\bold w}$ , corresponds with a linear transformation that moves the basis vectors i-hat and j-hat to $\vec{\bold v}$ and $\vec{\bold w}$ .

行列式就是变换前后**面积变化比例的度量**。

The determinant is all about **measuring how areas change** due to a transformation.

以 $\hat{\imath}$ 和 $\hat{\jmath}$ 为边的单位正方形，在变换之后，变成我们关心的平行四边形。

The unit square resting on i-hat and j-hat, after the transformation, that square gets turned into the parallelogram that we care about.

所以，通常用来度量面积变化比例的行列式在这里给出了**平行四边形的面积**。

So the determinant, which generally measures the factor by which areas are changed give **the area of this parallelogram**.

因为这个平行四边形来源于面积为 1 的正方形。

Since it evolved from a square that started with area 1.

### 三维空间

理解这个变换能够解释清楚**叉积的计算过程**和**几何含义**之间的关系。

Understanding that transformation is going to make clear the connection between **the computation** and **the geometry of the cross product**.

1. 根据 $\vec{\bold v}$ 和 $\vec{\bold w}$ 定义一个三维到一维的线性变换。

   Define a 3D-to-1D linear transformation in terms of $\vec{\bold v}$ and $\vec{\bold w}$ .

2. 找到它的对偶向量。

   Find its dual vector.

3. 说明这个对偶向量就是 $\vec{\bold v}\times \vec{\bold w}$ 。

   Show that this dual is $\vec{\bold v}\times \vec{\bold w}$ .



#### 寻找对偶向量

一个三维空间到数轴的函数。

A function from three dimensions to the number line.
$$
\large f\left(\begin{bmatrix}x \\y\\z\end{bmatrix}\right)=f\left(\begin{bmatrix}x \\y\\z\end{bmatrix}| \vec{\bold v}, \vec{\bold w}\right)=\rm{det}\left(\begin{bmatrix}x & v_1 & w_1 \\y & v_2 & w_2\\z & v_3 & w_3\end{bmatrix}\right)=V_{三个向量组成的平行六面体}
$$
这个函数是线性的，可以通过矩阵乘法来描述这个函数。

It's linear, there's some way to describe this function as matrix multiplication.
$$
\large f\left(\begin{bmatrix}x \\y\\z\end{bmatrix}\right)= \underbrace{\begin{bmatrix}?&?&?\end{bmatrix}}_{代表线性变换\\\ \ \  的1\times 3 矩阵}\begin{bmatrix}x \\y\\z\end{bmatrix}
$$
对偶性的整体思路是：

The whole idea  of duality is that :

从多维空间到一维空间的变换可以看作与这个特定向量的点积。

Interpret the transformations from several dimensions to one dimension as the dot product with a certain vector.
$$
\large f\left(\begin{bmatrix}x \\y\\z\end{bmatrix}\right)= \begin{bmatrix}?&?&?\end{bmatrix}\begin{bmatrix}x \\y\\z\end{bmatrix}=\begin{bmatrix}? \\?\\?\end{bmatrix}\cdot \begin{bmatrix}x \\y\\z\end{bmatrix}
$$
什么样的向量 $\vec{\bold p}$ 才能满足这一特殊性质：

What vector $\vec{\bold p}$ has the special property that : 

当你将向量 $\vec{\bold p}$ 和某个向量 $(x, y, z)$ 点乘时，所得结果等于一个 $3\times 3$ 矩阵的行列式。

When you take a dot product between $\vec{\bold p}$ and some vector $(x, y, z)$ , it gives the same result with the determinant of a $3 \times 3$ matrix.

这个矩阵第一列为 $(x, y, z)$ ，其余两列为 $\vec{\bold v}$ 和 $\vec{\bold w}$ 的坐标。

Plugging in (x, y, z) to the first column of the matrix, whose other two columns have  the coordinates of $\vec{\bold v}$ and $\vec{\bold w}$ .
$$
\large \begin{bmatrix}p_1 \\p_2\\p_3\end{bmatrix}\cdot \begin{bmatrix}x \\y\\z\end{bmatrix}=\rm{det}\left(\begin{bmatrix}x & v_1 & w_1 \\y & v_2 & w_2\\z & v_3 & w_3\end{bmatrix}\right)
$$

##### 数值方法

$$
\large \begin{array}
\ & x\left(v_{2} \cdot w_{3}-v_{3} \cdot w_{2}\right)+ \\
p_{1} \cdot x+p_{2} \cdot y+p_{3} \cdot z=& y\left(v_{3} \cdot w_{1}-v_{1} \cdot w_{3}\right)+ \\
&z\left(v_{1} \cdot w_{2}-v_{2} \cdot w_{1}\right)
\end{array}
$$

向量 $\vec{\bold p}$ 的坐标：

The coordinates of the vector $\vec{\bold p}$ :
$$
\large p_1 = v_{2} \cdot w_{3}-v_{3} \cdot w_{2}\\
\large p_2 = v_{3} \cdot w_{1}-v_{1} \cdot w_{3}\\
\large p_3 = v_{1} \cdot w_{2}-v_{2} \cdot w_{1}
$$

##### 几何方法

向量 $\vec{\bold p}$ 与其他向量的点积的几何解释是将其他向量投影到 $\vec{\bold p}$ 上，然后将投影长度与 $\vec{\bold p}$ 的长度相乘。

The geometric interpretation of a dot product between a vector $\vec{\bold p}$ and some other vector is to project that other vector onto $\vec{\bold p}$ , then to multiply the length of that projection by the length of $\vec{\bold p}$ .

我们可以推断出这个对偶向量必然与 $\vec{\bold v}$ 和 $\vec{\bold w}$ 垂直，并且其长度与这两个向量张成的平行四边形的面积相同。

We can deduce that this dual vector must be perpendicular to $\vec{\bold v}$ and $\vec{\bold w}$ , with a length equal to the area of parallelogram spanned out by those two vectors.



这两种方法给出了同一个变换的对偶向量，因此这两个向量必然相同。

Since both of these approaches give us a dual vector to the same transformation, they must be the same vector.

## 09 基变换

**我们视角下的向量与对方视角下的向量**
$$
我们的网格\quad \longrightarrow\quad \rm{Jennifer}的网格\\ \ \\
\large \begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}\\ \ \\
我们的语言\quad \longleftarrow\quad \rm{Jennifer}的语言\\
$$
理解：

$\longrightarrow$ ：这个矩阵所对应的线性变换把我们的网格，即我们的基向量转化为对方的基向量，用对方的基向量重新绘制网格。

$\longleftarrow$ ：这个矩阵和向量 $(x, y)$ 相乘，表示的是在对方网格中的这个向量 $(x, y)$ ，在我们的网格中应该怎么表述，即 $(x', y')$ 。
$$
\large \begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}\overbrace{\begin{bmatrix}x\\y\end{bmatrix}}^{对方的语言} = \underbrace{\begin{bmatrix}x'\\y'\end{bmatrix}}_{我们的语言}
$$
相反，
$$
\large \begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}^{-1} = \begin{bmatrix}1/3 & 1/3\\-1/3 & 2/3\end{bmatrix}
$$
将向量与这个基变换矩阵的逆相乘，
$$
\large \begin{bmatrix}1/3 & 1/3\\-1/3 & 2/3\end{bmatrix}\overbrace{\begin{bmatrix}x\\y\end{bmatrix}}^{我们的语言} = \underbrace{\begin{bmatrix}x'\\y'\end{bmatrix}}_{对方的语言}
$$


**我们视角下的线性变换与对方视角下的线性变换**

描述对方视角下某个向量 $(-1, 2)$ 经过我们视角下某种线性变换 $\begin{bmatrix}0 & -1\\1 & 0\end{bmatrix}$ 的过程：
$$
\large \overbrace{\begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}^{-1}}^{转换为用对方的\\语言描述的向量}\underbrace{\begin{bmatrix}0 & -1\\1 & 0\end{bmatrix}}_{用我们的语言描述\\的变换矩阵}\overbrace{\begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}}^{转换为用我们\\的语言描述}\underbrace{\begin{bmatrix}-1\\2\end{bmatrix}}_{对方的语言}
$$
对方视角下的线性变换：
$$
\large \begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}^{-1}\begin{bmatrix}0 & -1\\1 & 0\end{bmatrix}\begin{bmatrix}2 & -1\\1 & 1\end{bmatrix}=\begin{bmatrix}1/3 & -2/3\\5/3 & -1/3\end{bmatrix}
$$
我们视角下的线性变换 $\begin{bmatrix}0 & -1\\1 & 0\end{bmatrix}$ ，在对方的视角下应该是 $\begin{bmatrix}1/3 & -2/3\\5/3 & -1/3\end{bmatrix}$ 。



总的来说，每当你看到这样一个表达式： $A^{-1}MA$ ，这就暗示着一种数学上的转移作用。

In general, whenever you see an expression like A inverse times M times A , it suggests a mathematical sort of empathy.

中间的矩阵代表一种你所见的变换，而外侧两个矩阵代表着转移作用，也就是视角上的转化。

That middle matrix represents a transformation of some kind as you see it, and the outer two matrices represent the empathy, the shift in perspective.

所有矩阵的乘积仍然代表着同一个变换，只不过是从其他人的角度来看的。

And the full matrix product represents that same transformation, but as someone else sees it.

## 10 特征向量与特征值

对于一个线性变换而言，所有在变换后留在自己张成的空间里（方向不变）的向量就是特征向量。

特征向量在变换后方向不变，长度可能会发生缩放，缩放的倍数就是特征值。



**求解特征向量与特征值**

当且仅当矩阵代表的变换将空间压缩到更低的维度时，才会存在一个非零向量，使得矩阵和它的乘积为零向量，

而空间压缩对应的就是矩阵的行列式为零。
$$
\begin{align}
\large A\vec{\bold v}&=\lambda\vec{\bf v}\\
\large A\vec{\bold v}&=(\lambda I)\vec{\bf v}\\
\large (A-\lambda I)\vec{\bf v} &= \vec{\bf 0}\\
\large {\rm det} (A-\lambda I)&=0
\end{align}
$$


**矩阵的相似对角化**

线性变换 $\begin{bmatrix}3 & 1\\0 & 2\end{bmatrix}$ 的两个特征向量 $\begin{bmatrix}1\\0\end{bmatrix}$ 和 $\begin{bmatrix}-1\\1\end{bmatrix}$ ，作为一组特征基。
$$
\large \begin{bmatrix}1 & -1\\0 & 1\end{bmatrix}^{-1}\begin{bmatrix}3 & 1\\0 & 2\end{bmatrix}\underbrace{\begin{bmatrix}1 & -1\\0 & 1\end{bmatrix}}_{基变换矩阵}=\underbrace{\begin{bmatrix}3 & 0\\0 & 2\end{bmatrix}}_{特征空间视角下\\的线性变换}
$$
通过基变换矩阵把视角转移到特征空间中的视角，得到特征空间中的变换矩阵。

因为特征向量对于这个线性变换不会改变方向，即，只会对基向量作放缩，所以在特征空间中的这个变换矩阵一定是对角矩阵（对角元对应特征值）。

## 11 抽象向量空间

只要满足 “线性” 的要求，就可以运用线性代数的方法，不管处理的对象是否是向量。

线性：
$$
\large L(\vec{\bf v} + \vec{\bf w}) = L(\vec{\bf v}) + L(\vec{\bf w})\\
\large L(c \overrightarrow{\mathbf{v}})=c L(\overrightarrow{\mathbf{v}})
$$
“向量” 可以是 “函数” ，

对 “向量” 的 “线性变换” ，可以是对 “函数” 的 “线性算子” ，如 “求导” ：
$$
\large \frac{d}{d x}\left(x^{3}+x^{2}\right)=\frac{d}{d x}\left(x^{3}\right)+\frac{d}{d x}\left(x^{2}\right)\\
\large \frac{d}{d x}\left(4 x^{3}\right)=4 \frac{d}{d x}\left(x^{3}\right)
$$


“多项式” 可以是 “向量” ：
$$
\large a_{n} x^{n}+a_{n-1} x^{n-1}+\cdots a_{1} x+a_{0}=\left[\begin{array}{c}
a_{0} \\
a_{1} \\
\vdots \\
a_{n-1} \\
a_{n} \\
0 \\
\vdots
\end{array}\right]
$$
“求导” 可以是 “矩阵” ：
$$
\large \frac{d}{d x}=\left[\begin{array}{ccccc}
0 & 1 & 0 & 0 & \cdots \\
0 & 0 & 2 & 0 & \cdots \\
0 & 0 & 0 & 3 & \cdots \\
0 & 0 & 0 & 0 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{array}\right]
$$
求导过程：
$$
\begin{aligned}
&\frac{d}{d x}\left(1 x^{3}+5 x^{2}+4 x+5\right)=\underbrace{3 x^{2}+10 x+4}\\
&\left[\begin{array}{ccccc}
0 & 1 & 0 & 0 & \cdots \\
0 & 0 & 2 & 0 & \cdots \\
0 & 0 & 0 & 3 & \cdots \\
0 & 0 & 0 & 0 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots_{}
\end{array}\right]\left[\begin{array}{c}
5 \\
4 \\
5 \\
1 \\
\vdots
\end{array}\right]=\left[\begin{array}{c}
1 \cdot 4 \\
2 \cdot 5 \\
3 \cdot 1 \\
0 \\
\vdots
\end{array}\right]
\end{aligned}
$$

## 12 克莱姆法则

### 二维空间

解线性方程组
$$
\large \begin{bmatrix}2 & -1\\0 & 1\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix} = \begin{bmatrix}4\\2\end{bmatrix}
$$
基向量 $\vec{\bf \imath}$ 与向量 $(x, y)$ 围成的平行四边形面积
$$
\large {\rm Area} = 1 \times y = y
$$
基向量 $\vec{\bf \jmath}$ 与向量 $(x, y)$ 围成的平行四边形面积
$$
\large {\rm Area} = 1 \times x = x
$$
线性变换对应矩阵的行列式为面积放缩的比例
$$
\large {\rm newArea} = {\rm det}(A){\rm Area} = {\rm det}(A)y\\\ \\
\large y = \frac{\rm newArea}{{\rm det}(A)}\\\ \\
\large {\rm newArea} = {\rm det}(\begin{bmatrix}2 & 4\\0 & 2\end{bmatrix})\\\ \\
\large y = \frac{{\rm det}(\begin{bmatrix}2 & 4\\0 & 2\end{bmatrix})}{{\rm det}(A)}=\frac{{\rm det}(\begin{bmatrix}2 & 4\\0 & 2\end{bmatrix})}{{\rm det}(\begin{bmatrix}2 & -1\\0 & 1\end{bmatrix})}
$$
$\large x$ 同理。

### 三维空间

解线性方程组
$$
\large \begin{bmatrix}-4 & 2 & 3\\ -1 & 0 & 2\\-4 & 6 & -9\end{bmatrix}\begin{bmatrix}x\\y\\z\end{bmatrix} = \begin{bmatrix}7\\-8\\3\end{bmatrix}
$$
基向量 $\vec{\bf \imath}$ 和基向量 $\vec{\bf \jmath}$ 围成的单位正方形作为底面，与向量 $(x, y, z)$ 围成的平行六面体体积
$$
\large {\rm V} = 1 \times z = z
$$
线性变换对应矩阵的行列式为体积放缩的比例
$$
\large {\rm newV} = {\rm det}(A){\rm V} = {\rm det}(A)z\\\ \\
\large z = \frac{\rm newV}{{\rm det}(A)}\\\ \\
\large {\rm newV} = {\rm det}(\begin{bmatrix}-4 & 2 &7\\-1 & 0 & -8\\-4 & 6 & 3\end{bmatrix})\\\ \\
\large z = \frac{{\rm det}(\begin{bmatrix}-4 & 2 &7\\-1 & 0 & -8\\-4 & 6 & 3\end{bmatrix})}{{\rm det}(A)}=\frac{{\rm det}(\begin{bmatrix}-4 & 2 &7\\-1 & 0 & -8\\-4 & 6 & 3\end{bmatrix})}{{\rm det}(\begin{bmatrix}-4 & 2 & 3\\ -1 & 0 & 2\\-4 & 6 & -9\end{bmatrix})}
$$
$\large x$ 和 $\large y$ 同理。