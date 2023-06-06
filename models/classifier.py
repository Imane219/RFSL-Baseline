import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
import pdb

# @register('classifier_normalize_embedding')
# class classifier_normalize_embedding(nn.Module):
    
#     def __init__(self, encoder, encoder_args,
#                  classifier, classifier_args, metric='cos', temp=None):
#         super().__init__()
#         self.encoder = models.make(encoder, **encoder_args)
#         classifier_args['in_dim'] = self.encoder.out_dim
#         n_classes = classifier_args['n_classes']

#         self.weight = nn.Parameter(torch.empty(n_classes, classifier_args['in_dim']))
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

#         if temp is None:
#             if metric == 'cos':
#                 # temp = nn.Parameter(torch.tensor(10.))
#                 temp = 10.0
#             else:
#                 temp = 1.0
#         self.metric = metric
#         self.temp = temp

#     def forward(self, x):
#         x = self.encoder(x)
#         return utils.compute_logits(x, self.weight, self.metric, self.temp)

@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                # temp = nn.Parameter(torch.tensor(10.))
                temp = 10.0
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

