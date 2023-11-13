import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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



if __name__ == '__main__':
    datasets = ['wine', 'car', 'iris','adult']
    for dataset in datasets:
        print(f"Dataset:{dataset}")
        data_file = dataset + '.data'
        data_path = os.path.join('../datasets', dataset, data_file)
        X_train, X_test, y_train, y_test = read_data(dataset, data_path)
        svm_model = SVC(kernel='linear', C=1.0)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)