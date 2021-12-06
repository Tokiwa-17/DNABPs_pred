import numpy as np

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
    def __init__(self, train_positive, train_negative, nc=20, distance=3, cp=cp_20):
        # list of sequence
        self.train_positive = train_positive
        self.train_negative = train_negative
        self.nc = nc
        self.distance = distance
        self.cp = cp
        self.dimension = self.nc + self.nc ** 2 * self.distance
        self.train_positive_vector = np.zeros((len(train_positive), self.dimension))
        self.train_negative_vector = np.zeros((len(train_negative), self.dimension))

    def construct_training_matrix(self):
        for idx, seq in enumerate(self.train_positive):
            pseacc = PseAAC(seq, self.nc, self.distance, self.cp)
            self.train_positive_vector[idx] = np.array(pseacc.construct_feature_vector())

        for idx, seq in enumerate(self.train_negative):
            pseacc = PseAAC(seq, self.nc, self.distance, self.cp)
            self.train_negative_vector[idx] = np.array(pseacc.construct_feature_vector())

        print(f'training_dataset(pos) size: {self.train_positive_vector.shape}')
        print(f'training_dataset(neg) size: {self.train_negative_vector.shape}')


if __name__ == '__main__':
    train_positive = []
    with open('./dataset/PDB1075/sequence/positive.txt') as f:
        for line in f.readlines():
            if line[0] == '>':
                pass
            else:
                #print(line)
                train_positive.append(line.strip())
    train_negative = []
    with open('./dataset/PDB1075/sequence/negative.txt') as f:
        for line in f.readlines():
            if line[0] == '>':
                pass
            else:
                #print(line)
                train_negative.append(line.strip())

    nc = 20
    distance = 3
    cp = cp_20

    feature = Feature_extraction(train_positive, train_negative, nc, distance, cp)
    feature.construct_training_matrix()
    print("feature extract successfully")



