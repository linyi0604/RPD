
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score
import numpy as np  
import os
import torch

class Distribution_loss(torch.nn.Module):
    def __init__(self):
        super(Distribution_loss, self).__init__()
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, f1, f2, label):
        
        f1 = torch.log(torch.softmax(f1, dim=1))
        f2 = torch.softmax(f2, dim=1)

        class_features = {}
        class_target = {}
        
        for i in range(len(label)):
            l = label[i]
            if l not in class_features:
                class_features[l] = []
            if l not in class_target:
                class_target[l] = []

            class_features[l].append(f1[i])
            class_target[l].append(f2[i])

        loss = 0
        for l in class_features:
            loss += self.loss(torch.cat(class_features[l], dim=0), torch.cat(class_target[l], dim=0))

        return loss




class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def write(self, message):
        print(message.rstrip("\n"))
        with open(self.log_path, "a") as f:
            f.write(message + "\n")



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(predictions, labels, scores):
    # predictions = np.array(predictions)
    # labels = np.array(labels)
    # print(predictions.shape)
    # print(labels.shape)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    sensitivity = sensitivity_score(labels, predictions, average="macro")
    specificity = specificity_score(labels, predictions, average="macro")
    # raise
    if np.array(scores).shape[1] == 2:
        auc = roc_auc_score(labels, np.array(scores)[:, 1], average="macro", multi_class="ovr")
    else:
        auc = roc_auc_score(labels, scores, average="macro", multi_class="ovr")

    mean = np.mean([auc, accuracy, precision, specificity, sensitivity, f1])

    return auc, accuracy, precision, specificity, sensitivity, f1, mean