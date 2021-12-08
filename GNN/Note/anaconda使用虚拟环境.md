# 使用虚拟环境

### 创建虚拟环境并顺便装好ipykernel

```conda create -n env_name python=x.x ipykernel```

* ```env_name```为自定义的环境名称
* ```x.x```为想要安装的python版本号

 ### 激活环境

```conda activate env_name```

### 关闭环境

```conda deactivate```

### 删除环境

```conda remove -n env_name --all```

### 在环境中安装某个包

```conda install -n env_name [package]```

### 删除环境中的某个包

```conda remove --name $env_name  $package_name```

### 将环境写入Jupyter

```python -m ipykernel install --user --name $env_name --display-name "Python (环境名称)"```

### 意外问题

用Jupyter的notebook来import一些模块时可能出现

```
UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package
```

#### 解决

配置三个系统环境变量

```
\env_name
\env_name\Scripts
\env_name\Library\bin
```

将文件夹```\env_name\Library\bin```中的

```
libcrypto-1_1-x64.dll
libssl-1_1-x64.dll
```

复制到```\env_name\DLLs```文件夹中