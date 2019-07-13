import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from utils import read_image


class RetinaDataset(Dataset):
    """
    Creates a Dataset class to represent the retina image
    """

    def __init__(self, labels, directory, transform=None):

        self.labels = labels
        self.directory = directory
        self.size = size
        self.transform = transform
        super(RetinaDataset, self).__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels.iloc[index]

        img_name = "{}.png".format(label[0])
        img_label = label[1]

        image = read_image(self.directory, img_name)
        if self.transform:
            image = self.transform(image)

        return [image, img_label]


class RetinaDataLoader(object):
    """
    Creates a data-loader for training and testing data set
    """

    def __init__(self, tr_ds, te_ds=None, batch_size=16, shuffle=False):
        num_workers = os.cpu_count()

        self.tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)

        if te_ds is not None:
            self.te_dl = DataLoader(dataset=te_ds, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)

    def __getitem__(self, item):
        if item == 'train':
            return self.tr_dl
        elif item == 'test':
            return self.te_dl


class Transform(object):
    def __init__(self, resize=(224, 224)):
        list_of_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                 std=[0.485, 0.485, 0.485]),
            transforms.RandomSizedCrop(size=resize)
        ]
        self.transform = transforms.Compose(list_of_transforms)