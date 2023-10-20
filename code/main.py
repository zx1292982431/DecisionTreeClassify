import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from matplotlib import pyplot as plt

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

def train(X_train,y_train,criterion):
    tree_unpruned = DecisionTreeClassifier(criterion=criterion)
    tree_unpruned.fit(X_train, y_train)


    tree_prepruned = DecisionTreeClassifier(criterion=criterion, max_depth=3)
    tree_prepruned.fit(X_train, y_train)

    def prune_index(inner_tree, index, threshold):
        if inner_tree.value[index].min() < threshold:
            inner_tree.children_left[index] = _tree.TREE_LEAF
            inner_tree.children_right[index] = _tree.TREE_LEAF

        if inner_tree.children_left[index] != _tree.TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_left[index], threshold)
            prune_index(inner_tree, inner_tree.children_right[index], threshold)

    pruning_threshold = 0.05

    def prune_tree(tree, threshold):
        for index in range(tree.tree_.node_count):
            if tree.tree_.children_left[index] != tree.tree_.children_right[index]:
                left_child = tree.tree_.children_left[index]
                right_child = tree.tree_.children_right[index]

                info_gain = tree.tree_.impurity[index] - (
                            tree.tree_.weighted_n_node_samples[left_child] * tree.tree_.impurity[left_child] +
                            tree.tree_.weighted_n_node_samples[right_child] * tree.tree_.impurity[right_child]) / \
                            tree.tree_.weighted_n_node_samples[index]

                if info_gain < threshold:
                    tree.tree_.children_left[index] = tree.tree_.children_right[index] = -1

    tree_postpruned = DecisionTreeClassifier(criterion=criterion)
    tree_postpruned = tree_postpruned.fit(X_train, y_train)

    prune_tree(tree_postpruned, pruning_threshold)

    return tree_unpruned,tree_prepruned,tree_postpruned

def eval(tree_unpruned,tree_prepruned,tree_postpruned,X_test,y_test):
    acc_unpruned = accuracy_score(y_test, tree_unpruned.predict(X_test))
    acc_prepruned = accuracy_score(y_test,tree_prepruned.predict(X_test))
    acc_postpruned = accuracy_score(y_test,tree_postpruned.predict(X_test))
    return acc_unpruned,acc_prepruned,acc_postpruned

if __name__ == '__main__':
    datasets = ['adult','car','iris','wine']
    criterions = ['entropy','gini','log_loss']
    ans = {}
    for dataset in datasets:
        sub_ans = {}
        data_file = dataset+'.data'
        data_path = os.path.join('../datasets',dataset,data_file)
        X_train, X_test, y_train, y_test = read_data(dataset,data_path)
        for criterion in criterions:
            sub_sub_ans = {}
            tree_unpruned,tree_prepruned,tree_postpruned = train(X_train,y_train,criterion)
            sub_sub_ans['acc_unpruned'], sub_sub_ans['acc_prepruned'], sub_sub_ans['acc_postpruned'] = eval(tree_unpruned,tree_prepruned,tree_postpruned,X_test,y_test)
            sub_ans[criterion] = sub_sub_ans
        ans[dataset] = sub_ans

    print(ans)
    for i,dataset in enumerate(ans.keys()):
        plt.figure(figsize=(10, 15))
        for j,criterion in enumerate(ans[dataset].keys()):
            plt.subplot(len(ans[dataset].keys()),1,j+1)
            x = ans[dataset][criterion].keys()
            y = []
            for item in ans[dataset][criterion]:
                y.append(ans[dataset][criterion][item])
            print(y)
            plt.title(f'Criterion:{criterion}')
            plt.bar(x, y)
            for a, b, i in zip(x, y, range(len(x))):
                plt.text(a, b + 0.01, "%.2f" % y[i], ha='center', fontsize=10)
        plt.suptitle(f"Dataset:{dataset}")
        plt.savefig(f"../Ans/{dataset}_Ans.png",dpi=100)
        plt.show()



