import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from src.utils import read_image


class RetinaDataset(Dataset):
    """
    Creates a Dataset class to represent the retina image
    """

    def __init__(self, labels, directory, transform=None, test=False):

        self.labels = labels
        self.directory = directory
        self.transform = transform
        self.test = test
        super(RetinaDataset, self).__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels.iloc[index]

        img_name = "{}.png".format(label[0])

        image = read_image(self.directory, img_name)
        if self.transform is not None:
            image = self.transform(image)

        if self.test:
            return [image]

        img_label = label[1]
        return [image, img_label]


class RetinaDataLoader:
    """
    Creates a data-loader for training and testing data set
    """

    def __init__(self, tr_ds, val_ds=None, te_ds=None, batch_size=16, shuffle=False):
        num_workers = os.cpu_count()

        self.tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)

        self.val_dl = DataLoader(dataset=val_ds, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers)

        if te_ds is not None:
            self.te_dl = DataLoader(dataset=te_ds, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)

    def __getitem__(self, item):
        if item == 'train':
            return self.tr_dl
        elif item == 'val':
            return self.val_dl
        elif item == 'test':
            return self.te_dl


class Transform:
    def __init__(self, resize_to=(224, 224)):
        list_of_transforms = [
            transforms.Resize(size=resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(list_of_transforms)


class TransformTrainSet1:
    def __init__(self, resize_to=(224, 224)):
        list_of_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(list_of_transforms)


class TransformTrainSet2:
    def __init__(self, resize_to=(224, 224)):
        list_of_transforms = [
            transforms.RandomAffine(degrees=(-180, 180)),
            transforms.CenterCrop(size=resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(list_of_transforms)


class TransformTrainSet3:
    def __init__(self, resize_to=(224, 224)):
        list_of_transforms = [
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.Resize(size=resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(list_of_transforms)


class TransformTrainSet4:
    def __init__(self, resize_to=(224, 224)):
        list_of_transforms = [
            transforms.ColorJitter(brightness=0.35, contrast=0.15,
                                   saturation=0.10),
            transforms.Resize(size=resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(list_of_transforms)
