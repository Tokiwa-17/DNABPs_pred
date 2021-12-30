import numpy as np
from feature_selection.xgboost import Feature_selection_xgboost
from sklearn.model_selection import train_test_split
from kernel_SVM import SVM
from Random_Forest import RF

cp_13 = ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'T', 'S', 'N', 'RK', 'D', 'E']
cp_14 = ['EIMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'HRKQ', 'E', 'D']
cp_19 = ['P', 'G', 'E', 'K', 'R', 'Q', 'D', 'S', 'N', 'T', 'H', 'C', 'I', 'V', 'W', 'YF', 'A', 'L', 'M']
cp_20 = ['P', 'G', 'E', 'K', 'R', 'Q', 'D', 'S', 'N', 'T', 'H', 'C', 'I', 'V', 'W', 'Y', 'F', 'A', 'L', 'M']

class PseAAC:
    def __init__(self, sequence, nc=20, distance=3, cp=cp_20):
        self.sequence = sequence
        self.distance = distance
        self.nc = nc
        self.cp = cp
        self.dim = self.nc + self.nc ** 2 * self.distance
        self.length = len(sequence)
        self.vector = np.zeros(self.dim)
        #for residue in sequence:
        #    self.vector.append(self.cp.index(residue))
        #print(self.vector)

    def construct_feature_vector(self):
        for residue in self.sequence:
            try:
                self.vector[self.cp.index(residue)] += 1
            except:
                pass
        #print(self.vector[:20])
        self.vector /= self.length
        for i in range(2, self.distance + 2):
            for j in range(self.length):
                if j + i - 1 >= self.length:
                    break
                try:
                    residue_idx_left = self.cp.index(self.sequence[j])
                    residue_idx_right = self.cp.index(self.sequence[j + i - 1])
                    base_idx = self.nc + (self.nc ** 2) * (i - 2)
                    offset = residue_idx_left * self.nc + residue_idx_right
                    self.vector[base_idx + offset] += 1
                except:
                    pass
            self.vector[20 + (i - 2) * 400: 20 + (i - 1) * (400)] *= np.exp(i)

        return self.vector

class Feature_extraction:
    def __init__(self, positive, negative, nc=20, distance=3, cp=cp_20):
        # list of sequence
        self.positive = positive
        self.negative = negative
        self.nc = nc
        self.distance = distance
        self.cp = cp
        self.dimension = self.nc + self.nc ** 2 * self.distance
        self.positive_vector = np.zeros((len(positive), self.dimension))
        self.negative_vector = np.zeros((len(negative), self.dimension))
        self.positive_labels = np.ones(len(positive))
        self.negative_labels = -1 * np.ones(len(negative))
        self.dataset = None
        self.labels = None

    def construct_training_matrix(self):
        for idx, seq in enumerate(self.positive):
            pseacc = PseAAC(seq, self.nc, self.distance, self.cp)
            self.positive_vector[idx] = np.array(pseacc.construct_feature_vector())

        for idx, seq in enumerate(self.negative):
            pseacc = PseAAC(seq, self.nc, self.distance, self.cp)
            self.negative_vector[idx] = np.array(pseacc.construct_feature_vector())

        print(f'dataset(pos) size: {self.positive_vector.shape}')
        print(f'dataset(neg) size: {self.negative_vector.shape}')
        self.dataset = np.concatenate((self.positive_vector,
                                       self.negative_vector), axis=0)
        self.labels = np.concatenate((self.positive_labels, self.negative_labels))
        return self.dataset, self.labels

def read_dataset(filename):
    dataset = []
    with open(filename) as f:
        for line in f.readlines():
            if line[0]  == '>':
                pass
            else:
                dataset.append(line.strip())
    return dataset

if __name__ == '__main__':
    train_positive = read_dataset('./dataset/PDB1075/sequence/positive.txt')
    train_negative = read_dataset('./dataset/PDB1075/sequence/negative.txt')
    test_positive = read_dataset('./dataset/PDB186/sequence/positive.txt')
    test_negative = read_dataset('./dataset/PDB186/sequence/negative.txt')

    nc = 20
    distance = 3
    cp = cp_20

    # represent training dataset
    feature_train = Feature_extraction(train_positive, train_negative, nc, distance, cp)
    X_train, y_train = feature_train.construct_training_matrix()

    # split original training dataset to training dataset and validation dataset.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    # represent test dataset
    feature_test = Feature_extraction(test_positive, test_negative, nc, distance, cp)
    X_test, y_test = feature_test.construct_training_matrix()

    print("feature extract successfully")

    # feature selection
    feature_selection = Feature_selection_xgboost(X_train, y_train, X_test, y_test)
    feature_selection.compute_accuracy()

    X_train = feature_selection.select_features_by_thresh(X_train)
    X_test = feature_selection.select_features_by_thresh(X_test)

    # rbf kernel SVM classification
    svm = SVM(X_train, y_train, X_test, y_test)
    svm.predict()

    # random forest classification
    rf = RF(X_train, y_train, X_test, y_test)
    rf.predict()

