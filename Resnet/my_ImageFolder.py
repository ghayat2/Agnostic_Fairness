from torchvision import datasets
import ast


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
        return [item for sublist in [super().__getitem__(index), [w], [index]] for item in sublist]


class my_ImageFolderRandomCluster(datasets.ImageFolder):
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
        return [item for sublist in [super().__getitem__(index), [group_number], [index]] for item in sublist]


class my_ImageFolderCluster(datasets.ImageFolder):
    """
    This class redefines the ImageFolder class as to give a group id to each sample according to its demographic group
    """

    def __init__(self, root, transform, groups, clusters):
        super().__init__(root, transform)
        self.clusters = self.construct_clusters_1(clusters) if clusters else self.construct_clusters_2(groups)

    def __getitem__(self, index: int):
        return [item for sublist in [super().__getitem__(index), [self.clusters[index]], [index]] for item in sublist]

    def construct_clusters_1(self, dict_name):
        dic = read_dict(dict_name)
        clusters = []
        for path_img, label in self.samples:
            img = path_img.split("/")[-1]
            clusters.append(dic[img])

        return clusters

    def construct_clusters_2(self, groups):
        clusters = []
        for i, (path_img, label) in enumerate(self.samples):
            img = path_img.split("/")[-1]
            clusters.append(max([i if img in group else 0 for i, group in enumerate(groups)]))

        return clusters


def read_dict(clusters):
    with open(clusters) as f:
        data = f.read()
    return ast.literal_eval(data)
