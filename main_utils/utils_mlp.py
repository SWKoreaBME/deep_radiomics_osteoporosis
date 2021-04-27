# todo: Refactor this code

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def weights_init_he(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, bw):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(bw, 2, bias=True),
            nn.Dropout(0.8),
            nn.Softmax()
        )
        self.apply(weights_init_he)

    def forward(self, input):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        input = input.view(input.size(0), -1)
        input = self.layers(input)
        return input


class OsteoMLPDataset(torch.utils.data.Dataset):
    """Some Information about OsteoSiameseDataset"""

    def __init__(self, X, Y, transform, names=None):
        super(OsteoMLPDataset, self).__init__()

        self.transform = transform
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        input, label = self.X[index], self.Y[index]

        input = self.transform(input)
        label = self.transform(label)

        data = dict(
            input=input,
            label=label
        )

        return data

    def __len__(self):
        return len(self.X)


def torch2auc(output_list, label_list):
    from sklearn.metrics import auc, roc_curve

    label = np.zeros((len(label_list.numpy()), 2), dtype=int)
    for k, i in zip(label_list.numpy(), label):
        i[k] = 1

    output_list = output_list.numpy()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], thresholds[i] = roc_curve(label[:, i], output_list[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc[1]


def torch2confusion(output_list, label_list):
    output_list = torch.argmax(output_list, 1)
    confusion_result = confusion_matrix(label_list.numpy(), output_list.numpy())
    return confusion_result


def train_mlp(train_loader, valid_loader, bw=256, epochs=100, device="cuda"):
    model = MLP(bw)
    model = model.to(device)
    # model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    valid_auc_list = []

    best_auc = 0.0

    for epoch in tqdm(range(epochs)):
        model.train()

        train_losses = []
        valid_losses = []

        # train loop
        for i, data in enumerate(train_loader):

            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        output_list = torch.tensor([])
        label_list = torch.tensor([]).long()

        scheduler.step()

        with torch.no_grad():
            for i, data in enumerate(valid_loader):

                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                output_list = torch.cat((output_list, outputs.detach().cpu()))
                label_list = torch.cat((label_list, labels.detach().cpu()))

        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))

        accuracy = 100 * correct / total
        valid_acc_list.append(accuracy)

        auc = torch2auc(output_list, label_list)
        valid_auc_list.append(auc)

        if auc >= best_auc:
            best_auc = auc

    print('>> epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%, valid auc : {:.2f}' \
          .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), accuracy, auc))

    print('>> best auc:', best_auc)

    print(torch2confusion(output_list, label_list))

    plt.plot(mean_valid_losses)
    plt.plot(list(np.array(valid_acc_list) / 100))
    plt.plot(valid_auc_list)
    plt.legend(['loss', 'acc', 'auc'])
    plt.title('Validation Result')
    plt.show()

    return model


def save_infer_result(output,
                      version,
                      subjects,
                      dataset_type,
                      model_type,
                      save_path='./'):
    preds = output["preds"]
    output_list = output["output_list"]

    ext_result = np.hstack(
        [np.expand_dims(preds, axis=1), output_list])

    external_result = pd.DataFrame(ext_result, index=subjects,
                                   columns=['label', 'confidence score for class 0',
                                            'confidence score for class 1'])
    os.makedirs(os.path.join(save_path, dataset_type), exist_ok=True)
    external_result.to_csv(
        os.path.join(save_path, dataset_type, '{}_result_{}_{}.csv'.format(model_type,
                                                                           version,
                                                                           dataset_type)))


def infer_mlp(model, valid_loader, device="cuda"):
    # Get already wrapped model
    # if torch.cuda.is_available():
    #     model = model.cuda()
    model = model.to(device)

    model.eval()
    output_list = torch.tensor([])
    with torch.no_grad():
        for inputs in tqdm(valid_loader):
            if torch.cuda.is_available():
                inputs = inputs.to(device)
            outputs = model(inputs)
            output_list = torch.cat((output_list, outputs.detach().cpu()))
    preds = torch.argmax(output_list, 1).detach().cpu().numpy()
    return dict(
        preds=preds,
        output_list=output_list.detach().numpy()
    )


if __name__ == '__main__':
    pass
