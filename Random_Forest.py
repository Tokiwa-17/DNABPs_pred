from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

N_ESTIMATORS = 200

class RF:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # 对n_estimators调参 n_estimators = 121
        '''score_list = []
        for i in range(0, N_ESTIMATORS + 1, 10):
            rfc = RandomForestClassifier(n_estimators=i + 1
                                         , random_state=90)
            score = cross_val_score(rfc, self.X_train, self.y_train, cv=10).mean()
            score_list.append(score)
        score_max = max(score_list)
        print('最大得分：{}'.format(score_max),
              '子树数量为：{}'.format(score_list.index(score_max) * 10 + 1))

        # 绘制学习曲线
        x = np.arange(0, N_ESTIMATORS + 1, 10)
        plt.subplot(111)
        plt.plot(x, score_list, 'r-')
        plt.show()
        '''
        # 对max_depth调参
        rfc = RandomForestClassifier(n_estimators=121, random_state=90)
        param_grid = {'max_depth': np.arange(1, 20)}
        grid_search_2 = GridSearchCV(rfc, param_grid, cv=10)
        grid_search_2.fit(self.X_train, self.y_train)
        best_param = grid_search_2.best_params_
        best_score = grid_search_2.best_score_
        print(f'best_param:{best_param}, best_score:{best_score}')

        self.rf = RandomForestClassifier(n_estimators=121)
        self.rf.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.rf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy of using Random Forest:{accuracy}')
