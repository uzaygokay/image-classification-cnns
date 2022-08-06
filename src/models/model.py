#%% imports 
from sys import prefix
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, Accuracy, F1Score

from alexnet import AlexNet
from inceptionnet import InceptionModule
#%% model class
class ConvModel(LightningModule) :
    def __init__(self, channels, width, height, num_classes, hidden_size, learning_rate, weight_decay, dropout):
        super().__init__()

        #save hyperparameters
        self.save_hyperparameters()

        #model arguments
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout

        #define metrics
        metrics = MetricCollection({

            'accuracy' : Accuracy(),
            'f1_micro' : F1Score(num_classes=self.num_classes, average='micro'),
            'f1_macro' : F1Score(num_classes=self.num_classes, average='macro')
        }

        )

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')


        #model
        #self.model = nn.Sequential(
        #    nn.Flatten(),
        #    nn.Linear(channels * width * height, self.hidden_size),
        #    nn.ReLU(),
        #    nn.Dropout(self.dropout),
        #    nn.Linear(self.hidden_size, self.hidden_size),
        #    nn.ReLU(),
        #    nn.Dropout(self.dropout),
        #    nn.Linear(self.hidden_size, self.num_classes)
        #)

        #alexnet
        self.model = AlexNet(num_classes=self.num_classes, dropout=self.dropout)
        
        #inceptionnet
        #self.model = InceptionModule(in_channels=3,out_channels=32)

        #googlenet pretrained
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

    def forward(self, x):
        #call model and apply softmax
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x,y = batch 
        logits = self(x)
        
        #calculate loss and update metrics
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, y)
        
        #log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss
    
    def training_epoch_end(self, outputs):
        #compute the metrics and reset
        result = self.train_metrics.compute()
        self.train_metrics.reset()

        #log the metrics
        self.log_dict(result, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x,y = batch 
        logits = self(x)
        
        #calculate loss and update metrics
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_metrics.update(preds, y)

        #log loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        #compute the metrics and reset
        result = self.valid_metrics.compute()
        self.valid_metrics.reset()

        #log the metrics
        self.log_dict(result, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x,y = batch 
        logits = self(x)
        
        #calculate loss and update metrics
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, y)

        #log loss
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs):
        #compute the metrics and reset
        result = self.test_metrics.compute()
        self.test_metrics.reset()

        #log the metrics
        self.log_dict(result, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    

    

