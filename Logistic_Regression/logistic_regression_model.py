import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, num_predictor_features):
        super(Predictor, self).__init__()
        self.linear = torch.nn.Linear(num_predictor_features, 1)

    def forward(self, x):
        y_logits = self.linear(x)
        y_pred = F.sigmoid(y_logits)
        return y_logits, y_pred


def train(model, device, train_loader, optimizer, verbose=1):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0

    batches = enumerate(train_loader)

    for batch_idx, (data, target, protect) in batches:
        data, target, protect = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float), protect.to(
            device, dtype=torch.float)
        optimizer.zero_grad()
        logits, output = model(data)
        criterion = torch.nn.BCELoss()
        loss = criterion(output, target.view_as(output))
        pred = (output > 0.5) * 1
        pred = pred.float()
        correct = pred.eq(target.view_as(pred)).sum().item()
        sum_num_correct += correct
        sum_loss += loss.item() * train_loader.batch_size
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()

    sum_loss /= len(train_loader.dataset)
    train_accuracy = sum_num_correct / len(train_loader.dataset)

    if verbose:
        print('\nTrain set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            sum_loss, sum_num_correct, len(train_loader.dataset),
            100. * sum_num_correct / len(train_loader.dataset)))

    return sum_loss, train_accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_pred = torch.zeros(0, 1).to(device)
    with torch.no_grad():
        for data, target, protect in test_loader:
            data, target, protect = data.to(device, dtype=torch.float), target.to(device,
                                                                                  dtype=torch.float), protect.to(device,
                                                                                                                 dtype=torch.float)
            logit, output = model(data)
            criterion = torch.nn.BCELoss()
            loss = criterion(output, target.view_as(output))
            test_loss += loss.item() * test_loader.batch_size  # sum up loss for each test sample
            pred = (output > 0.5) * 1
            pred = pred.float()
            test_pred = torch.cat([test_pred, pred], 0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    return test_pred, test_loss, test_accuracy
