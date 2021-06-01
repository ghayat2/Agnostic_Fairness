from torchvision import datasets

class my_ImageFolder(datasets.ImageFolder):
    """
    This class redefines the ImageFolder class as the weight of each image is returned along with the data and label
    """
    def __init__(self, root, transform, protected_group, w_protected):
        super().__init__(root, transform)
        self.protected_group = protected_group
        self.w_protected = w_protected

    def __getitem__(self, index: int):
        w = self.w_protected if self.samples[index][0].split("/")[-1] in self.protected_group else 1
        return super().__getitem__(index), w #index


class my_ImageFolderCluster(datasets.ImageFolder):
    """
    This class redefines the ImageFolder class as the group of each image is returned along with the data and label
    """
    def __init__(self, root, transform, clusters):
        super().__init__(root, transform)
        self.clusters = clusters

    def __getitem__(self, index: int):
        img = self.samples[index][0].split("/")[-1]
        group_number = max(
            [[img in c for c in clusters].index(max([img in c for c in clusters])) for clusters in self.clusters])
        return super().__getitem__(index), group_number
