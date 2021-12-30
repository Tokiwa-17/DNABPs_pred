from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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
        self.clf_rbf = SVC(kernel='rbf', C=1, gamma=10)
        self.clf_rbf.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.clf_rbf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy of using rbf kernel SVM:{accuracy}')



