# test-100day-MachineLearning

机器学习 测试练习 
python版本: Python 3.7.0

参考 [机器学习100天 - 中文版](https://github.com/MLEveryday/100-Days-Of-ML-Code)

`Jupyter Notebook` 设置变量不使用`print`直接打印:

 - 仅更改当前页面: 

   ```python
   from IPython.core.interactiveshell import InteractiveShell
   InteractiveShell.ast_node_interactivity = "all"
   ```

 - 永久更改: 
    新建 `~/.ipython/profile_default/ipython_config.py` 文件并写入一下内容：

  ```python
  c = get_config()
  c.InteractiveShell.ast_node_interactivity = "all"  # Run all nodes interactively
  ```

  

Day1: 数据预处理
---------------------
1) "pd.read_csv()"  需要添加参数 "index_col"，默认第一列为index，测试数据没有，设为"false" \
   pd.read_csv(data_path, index_col=False)

2) step5 拆分数据为训练集和测试集: 使用 "from sklearn.model_selection import train_test_split" 



## Day2: 简单线性回归

简单线性回归是确定线性函数的斜率和截距，使得误差最小化（就是下图中同一个x值对应的实际的y值与直线对应y值的偏差平方和的值最小)

![02_LinearRegression](E:\code\jupyter\test-100day-MachineLearning\picture\02_LinearRegression.png)

观测值: $(Hours,Score) = \{(x_i, y_i), i=1,2,...,n\}$, 确定函数 $y=b + k * x$， 

使得$\sum{(y_i-y_i')^2}$的值最小，其中, $y_i'=b + k * x_i$.

相关系数$R^2$为 $R^2 = 1 - \frac{u}{v}$, 其中，

$u = \sum(y_i - y_i')^2$, $v = \sum(y_i - \bar{y})^2$, $\bar{y} = \frac{\sum{y_i}}{n}$

> The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() 
> and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()

