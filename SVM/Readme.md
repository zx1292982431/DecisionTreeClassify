### 作业3实验报告
#### 0201121713 李子轩 信息管理与信息系统
#### 数据集
本实验选用UCI数据集中选用了以下四个数据集进行实验：
* Adult
* Car 
* Iris
* Wine
#### 实验环境
1. Python==3.8
2. pands==1.5.3
3. scikit-learn==1.2.1
#### 实验方法
1. 数据读取与预处理

    首先对数据集进行读取，并进行一定的处理，将数据集中的非数值类转为数值类，以便于决策树模型拟合数据。
    
    在进行预处理后通过 `sklearn.model_selection` 划分训练集和测试集

    本实验中定义了 `read_data` 方法来读取、预处理、划分数据集，具体代码如下：

    ```python
    def read_data(dataset,path):
    data = pd.read_csv(path, header=None)
   
    if dataset=='adult':
        column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
                        "native_country", "income"]
   
        adult_data = pd.read_csv(path, names=column_names, skipinitialspace=True)
   
        categorical_columns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex",
                               "native_country"]
        adult_data_encoded = pd.get_dummies(adult_data, columns=categorical_columns, drop_first=True)
   
        X = adult_data_encoded.drop("income", axis=1)
        y = adult_data_encoded["income"]
   
    elif dataset=='car':
        column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        car_data = pd.read_csv(path, names=column_names)
   
        categorical_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        car_data_encoded = pd.get_dummies(car_data, columns=categorical_columns)
   
        X = car_data_encoded.drop("class", axis=1)
        y = car_data_encoded["class"]
   
    elif dataset=='iris':
        column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        iris_data = pd.read_csv(path, names=column_names)
   
        X = iris_data.drop("species", axis=1)
        y = iris_data["species"]
   
    elif dataset =='wine':
        wine_data = pd.read_csv(path, header=None)
   
        X = wine_data.iloc[:, 1:]
        y = wine_data.iloc[:, 0]
   
    else:
        print("不支持的数据集")
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    return X_train, X_test, y_train, y_test
    ```
    2. 模型定义与训练

    本次实验借助`sklearn`库中线性核的SVM进行实验。
    ```python
    svm_model = SVC(kernel='linear', C=1.0)
    ```
    使用SVC类的Train接口进行训练
    ```python
    svm_model.fit(X_train, y_train)
    ```
    3. 模型评估

    本实验借助`sklearn.metrics`中的`accuracy_score`和`classification_report`接口进行评估
    ```python
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    ```
    4. 实验结果

    Wine数据集：
    ```
    Accuracy: 1.00
    Classification Report:
              precision    recall  f1-score   support
   
           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00        14
           3       1.00      1.00      1.00         8
   
        accuracy                       1.00        36
        macro avg  1.00      1.00      1.00        36
        weighted   1.00      1.00      1.00        36
        avg
    ```
    Car数据集：
    ```
    Dataset:car
    Accuracy: 0.93
    Classification Report:
              precision    recall  f1-score   support
   
         acc       0.92      0.80      0.85        83
        good       0.50      0.91      0.65        11
       unacc       0.97      0.97      0.97       235
       vgood       0.94      1.00      0.97        17
   
        accuracy                       0.93       346
    macro avg      0.83      0.92      0.86       346
    weighted avg   0.94      0.93      0.93       346
    ```
    Iris数据集：
    ```
    Accuracy: 1.00
    Classification Report:
                 precision    recall  f1-score   support
   
    Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor   1.00      1.00      1.00         9
    Iris-virginica    1.00      1.00      1.00        11
   
       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg      1.00      1.00      1.00        30
   ```
    Adult数据集：
    ```
    Accuracy: 0.80
    Classification Report:
              precision    recall  f1-score   support
   
       <=50K       0.81      0.95      0.88      4942
        >50K       0.68      0.31      0.43      1571
   
    accuracy                           0.80      6513
    macro avg      0.75      0.63      0.65      6513
    weighted avg   0.78      0.80      0.77      6513
    ```