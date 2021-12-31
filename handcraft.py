hydrophobic_amino_acids = ['G', 'A', 'V', 'L', 'I', 'P', 'F']
hydrophilic_amino_acids = ['Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', '']
negative_amino_acids = ['D', 'E']
positive_amino_acids = ['K', 'R', 'H']

class Handcraft:
    def __init__(self, sequence):
        """

        :param sequence: protein amino acids sequence
        """
        self.sequence = sequence
        self.hydrophobic_rich = False
        self.hydrophilic_rich = False
        self.negative_rich = False
        self.positive_rich = False

    def get_attribute(self):
        for i in range(len(self.sequence) - 10):
            hydrophobic_cnt = 0
            hydrophilic_cnt = 0
            negative_cnt = 0
            positive_cnt = 0
            for j in range(i, i + 10):
                if self.sequence[j] in hydrophobic_amino_acids:
                    hydrophobic_cnt += 1
                    if hydrophobic_cnt >= 6:
                        self.hydrophobic_rich = True
                elif self.sequence[j] in hydrophilic_amino_acids:
                    hydrophilic_cnt += 1
                    if hydrophilic_cnt >= 6:
                        self.hydrophilic_rich = True
                elif self.sequence[j] in negative_amino_acids:
                    negative_cnt += 1
                    if negative_cnt >= 4:
                        self.negative_rich = True
                elif self.sequence[j] in positive_amino_acids:
                    positive_cnt += 1
                    if positive_cnt >= 4:
                        self.positive_rich = True

        return  int(self.hydrophobic_rich), int(self.hydrophilic_rich), int(self.positive_rich), int(self.negative_rich)

