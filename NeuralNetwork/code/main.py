import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Dataset import MyDataset
from utils import split_train_and_test
from Parameters import Parameters
from tqdm import tqdm
from Model import MLP,MLP_mini
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ['adult','car','iris','wine']
    in_dim = {
        'adult': 14,
        'car': 6,
        'iris': 4,
        'wine': 13
    }
    out_dim ={
        'adult': 2,
        'car': 4,
        'iris': 3,
        'wine': 3
    }
    for dataset in datasets:
        parameters = Parameters()
        file_name = dataset + '.data'
        data_path = os.path.join('../datasets',dataset,file_name)
        split_train_and_test(data_path)
        train_data_path = os.path.join('../datasets',dataset,'train.data')
        test_data_path = os.path.join('../datasets',dataset,'test.data')
        train_dataset = MyDataset(dataset,train_data_path)
        test_dataset = MyDataset(dataset,test_data_path)
        train_loader = DataLoader(train_dataset,batch_size=parameters.train_batch,shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=parameters.test_batch)

        # 可选MLP与MLP_mini
        model = MLP(in_dim[dataset],out_dim[dataset])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=parameters.lr)

        train_loss_record=[]
        eval_loss_record=[]
        acc_record =[]

        for epoch in tqdm(range(parameters.epochs)):
            model.train()
            train_loss = 0.0
            for feature,label in train_loader:
                feature = feature.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.int64)
                optimizer.zero_grad()
                output = model(feature)
                loss = criterion(output,label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_loss_record.append(train_loss)

            model.eval()
            eval_loss = 0.0
            total = 0
            correct = 0
            with torch.no_grad():
                for feature,label in test_loader:
                    feature = feature.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.int64)
                    output = model(feature)
                    loss = criterion(output,label)
                    _,predicted = torch.max(output, 1)
                    eval_loss += loss.item()
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                acc = correct / total
                acc_record.append(acc)
                total = 0
                correct = 0
            eval_loss /= len(test_loader)
            eval_loss_record.append(eval_loss)

        plt.figure(figsize=(12, 4))
        plt.title(f'Dataset:{dataset}')
        plt.subplot(1, 2, 1)
        plt.plot(range(1, parameters.epochs + 1), train_loss_record, label='Train Loss')
        plt.plot(range(1, parameters.epochs + 1), eval_loss_record, label='Eval Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, parameters.epochs + 1), acc_record, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


