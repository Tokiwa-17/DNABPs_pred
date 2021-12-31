import numpy as np
import math

class Evaluation:
    def __init__(self, pred, labels):
        self.length = pred.shape[0]
        self.pred = pred
        self.labels = labels
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.Sn = 0
        self.Sp = 0
        self.Acc = 0
        self.MCC = 0

    def compute_metrics(self):
        for i in range(self.length):
            if self.pred[i] == 1 and self.labels[i] == 1:
                self.TP += 1
            elif self.pred[i] == -1 and self.labels[i] == -1:
                self.TN += 1
            elif self.pred[i] == 1 and self.labels[i] == -1:
                self.FP += 1
            else:
                self.FN += 1

        self.Sn = self.TP / (self.TP + self.FN)
        self.Sp = self.TN / (self.TN + self.FP)
        self.Acc = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.MCC = (self.TP * self.TN - self.FP * self.FN) / math.sqrt((self.TP + self.FP) * (self.TP + self.FN) *
                                                                       (self.TN + self.FP) * (self.TN + self.FN))
        print(f'Evaluation Metrics: Sn:{self.Sn}, Sp:{self.Sp}, Acc:{self.Acc}, MCC:{self.MCC}')
