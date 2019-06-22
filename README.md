# test-100day-MachineLearning

机器学习 测试练习 \
python版本: Python 3.7.0

Day1: 数据预处理
---------------------
1) "pd.read_csv()"  需要添加参数 "index_col"，默认第一列为index，测试数据没有，设为"false" \
   pd.read_csv(data_path, index_col=False)
   
2) step5 拆分数据为训练集和测试集: 使用 "from sklearn.model_selection import train_test_split" 
