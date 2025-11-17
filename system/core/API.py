import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_data

class train(object):
    def __init__(self, args, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.dataset = args.dataset
        self.device = args.device
        self.save_folder_name = args.save_folder_name
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        
        self.learning_rate_decay = args.learning_rate_decay
    
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        train_data = read_data(self.dataset, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    
    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        test_data = read_data(self.dataset, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
    
    def test_metrics(self):
        self.model.eval()
        y_true = []
        y_score = []

        with torch.no_grad():
            for batch in self.load_test_data():
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probabilities = nn.Softmax(dim=1)(outputs)

                y_true.extend(labels.cpu().numpy())
                y_score.extend(probabilities.cpu().numpy())

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        test_acc = metrics.accuracy_score(y_true, np.argmax(y_score, axis=1))
        y_true_binarized = label_binarize(y_true, classes=list(range(self.num_classes)))
        test_auc = metrics.roc_auc_score(y_true_binarized, y_score, average='macro', multi_class='ovr')

        return test_acc, test_auc
    
    def train_metrics(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.load_train_data():
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss