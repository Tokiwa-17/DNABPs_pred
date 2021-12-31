from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from evaluation import Evaluation
from draw.ROC import ROC
import numpy as np
from sklearn.model_selection import GridSearchCV

class SVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # adjust parameter
        # grid = GridSearchCV(SVC(), param_grid={"C": [0.01, 0.1, 0.5, 1, 3, 5, 10, 100], "gamma": [50, 20, 15, 10, 1, 0.1, 0.01, 0.001]}, cv=4)
        # grid.fit(X_train, y_train)
        # best_parameters = grid.best_estimator_.get_params()
        # print("The best parameters are %s with a score of %0.2f"
        #      % (grid.best_params_, grid.best_score_))
        #self.clf_rbf = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        self.clf_rbf_1 = SVC(kernel='rbf', C=1, gamma=10, probability=True)
        self.clf_rbf_2 = SVC(kernel='rbf', C=10, probability=True)
        self.clf_rbf_3 = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
        self.clf_rbf_4 = SVC(kernel='rbf', probability=True)
        self.clf_rbf_5 = SVC(kernel='rbf', gamma=100, probability=True)
        self.clf_rbf_1.fit(self.X_train, self.y_train)
        self.clf_rbf_2.fit(self.X_train, self.y_train)
        self.clf_rbf_3.fit(self.X_train, self.y_train)
        self.clf_rbf_4.fit(self.X_train, self.y_train)
        self.clf_rbf_5.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred_1 = self.clf_rbf_1.predict(self.X_test)
        y_pred_2 = self.clf_rbf_2.predict(self.X_test)
        y_pred_3 = self.clf_rbf_3.predict(self.X_test)
        y_pred_4 = self.clf_rbf_4.predict(self.X_test)
        y_pred_5 = self.clf_rbf_5.predict(self.X_test)
        y_pred =  y_pred_1 * 1.5 + y_pred_2 + y_pred_3 + y_pred_4 + y_pred_5
        y_pred = np.where(y_pred > 0, 1.0, -1.0)
        evaluation = Evaluation(y_pred, self.y_test)
        evaluation.compute_metrics()
        y_pred_prob = self.clf_rbf_1.predict_proba(self.X_test) + self.clf_rbf_2.predict_proba(self.X_test) + \
        self.clf_rbf_3.predict_proba(self.X_test) + self.clf_rbf_4.predict_proba(self.X_test) + self.clf_rbf_5.predict_proba(self.X_test)
        y_pred_prob /= 5

        roc = ROC(y_pred_prob[:, 1], self.y_test)
        roc.draw()
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy of using rbf kernel SVM:{accuracy}%\n')



