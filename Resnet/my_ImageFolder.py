from torchvision import datasets


class my_ImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, protected_group, w_protected):
        super().__init__(root, transform)
        self.protected_group = protected_group
        self.w_protected = w_protected

    def __getitem__(self, index: int):
        w = self.w_protected if self.samples[index][0].split("/")[-1] in self.protected_group else 1
        return super().__getitem__(index), w
