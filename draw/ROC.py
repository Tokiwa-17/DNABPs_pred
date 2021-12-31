from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ROC:
    def __init__(self, y_pred, y_label):
        self.y_pred = y_pred
        self.y_label = y_label
        self.fpr, self.tpr, self.thrsh = roc_curve(y_label, y_pred, pos_label=1)
        self.roc_auc = auc(self.fpr, self.tpr)

    def draw(self):
        plt.plot(self.fpr, self.tpr, label='ROC (area = {0:.2f})'.format(self.roc_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()