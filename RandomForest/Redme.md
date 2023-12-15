### 作业4实验报告
#### 0201121713 李子轩 信息管理与信息系统
#### 数据集
本实验选用UCI数据集中选用了鸢尾花数据集进行实验：
* Iris
#### 实验环境
1. Python==3.8
2. numpy==1.23.1
3. scikit-learn==1.2.1
5. matplotlib==3.7.1
#### 实验方法
1. 数据读取与预处理

    ```python
   data = load_iris()
    X = data.data
    y = data.target
   ```
   本实验通过sklean.dataset中读取鸢尾花数据集的`load_iris`接口进行数据集的读取

2. 随机森林模型

    ```python
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    ```
   本实验借助sklearn提供的`sklearn.ensemble.RandomForestClassifier`类创建随机森林分类器
3. 模型训练与评估
    
    ```python
    rf_classifier.fit(X_train, y_train)
   ```
   本实验借助RandomForestClassifier.fit接口在鸢尾花数据集上进行训练
#### 实验结果

    Accuracy: 1.00
    Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
    macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30

随机森林算法在各个类别上均取得了100%的Acc
