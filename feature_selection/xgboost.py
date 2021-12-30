from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import numpy as np

class Feature_selection_xgboost:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.model = XGBClassifier()
        self.thresholds = None
        self.selected_thresh = 0.000779

    def compute_accuracy(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f'Accuracy of using XGboost: {accuracy * 100.0}%')
        # print(f'importance of standard amino acids:{self.model.feature_importances_[:20]}')
        self.thresholds = np.sort(self.model.feature_importances_)

        return accuracy

    def select_features(self):
        #thresholds = np.sort(self.model.feature_importances_)
        print(np.where(self.thresholds == 0)[0].shape[0])
        zero_feature_number = np.where(self.thresholds == 0)[0].shape[0]
        thresholds_list = []
        for i in range(zero_feature_number, self.X_train[0].shape[0], 50):
            thresholds_list.append(self.thresholds[i])
        #print(thresholds_list)
        #thresh = self.thresholds[zero_feature_number]

        for thresh in thresholds_list:
            # select features using threshold
            selected_features = SelectFromModel(self.model, threshold=thresh, prefit=True)
            # reduce X to the selected features
            select_X_train = selected_features.transform(self.X_train)
            # train model
            new_model = XGBClassifier()
            new_model.fit(select_X_train, self.y_train)
            # eval model
            select_X_test = selected_features.transform(self.X_test)
            y_pred = selected_features.transform(self.X_test)
            y_pred = new_model.predict(select_X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f'Threshold: {thresh}, feature_number: {select_X_train.shape[1]}, '
                  f'Accuracy: {accuracy * 100.0}%')
        #selected_features = SelectFromModel(self.model, threshold=self.selected_thresh, prefit=True)
        #return selected_features.transform(self.X_train), selected_features.transform(self.X_test)

    def select_features_by_thresh(self, vec):
        # return reduced features
        selected_features = SelectFromModel(self.model, threshold=self.selected_thresh, prefit=True)
        return selected_features.transform((vec))


