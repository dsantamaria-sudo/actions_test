from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.datasets import Food101


class _RemapLabel:
    """Picklable label remapper (a lambda would break num_workers>0 on Windows)."""

    def __init__(self, old_to_new):
        self.old_to_new = old_to_new

    def __call__(self, label):
        return self.old_to_new[label]


def create_dataloaders(train_dir, test_dir, train_transform, test_transform, BATCH_SIZE, test_batch_size=None, num_workers=2):

    train_data = datasets.ImageFolder(root=train_dir,
                                    transform=train_transform,
                                    target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=test_transform,
                                    target_transform=None)

    class_names = train_data.classes
    class_dict = train_data.class_to_idx

    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                num_workers=num_workers,
                                shuffle=True)

    # Default to a single batch covering the whole test set: with per-batch
    # accuracy averaged across batches, a small/uneven batch count quantizes
    # the metric into a handful of repeated values instead of a true correct/total.
    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=test_batch_size or len(test_data),
                                num_workers=num_workers,
                                shuffle=False)

    return train_dataloader, test_dataloader, class_names, class_dict


def create_food101_dataloaders(data_dir, classes, train_transform, test_transform, BATCH_SIZE,
                                test_batch_size=None, num_workers=2, download=True):
    """Same return shape as create_dataloaders, but pulls `classes` out of the
    full Food-101 dataset (750 train / 250 test images per class) instead of
    the local ImageFolder mini-subset."""

    full_train = Food101(root=data_dir, split="train", download=download)
    full_test = Food101(root=data_dir, split="test", download=download)

    missing = [c for c in classes if c not in full_train.class_to_idx]
    if missing:
        raise ValueError(f"Classes not found in Food101: {missing}. Available: {full_train.classes}")

    old_to_new = {full_train.class_to_idx[c]: i for i, c in enumerate(classes)}
    remap = _RemapLabel(old_to_new)

    full_train.transform = train_transform
    full_train.target_transform = remap
    full_test.transform = test_transform
    full_test.target_transform = remap

    train_indices = [i for i, label in enumerate(full_train._labels) if label in old_to_new]
    test_indices = [i for i, label in enumerate(full_test._labels) if label in old_to_new]

    train_data = Subset(full_train, train_indices)
    test_data = Subset(full_test, test_indices)

    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                num_workers=num_workers,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=test_batch_size or len(test_data),
                                num_workers=num_workers,
                                shuffle=False)

    class_dict = {c: i for i, c in enumerate(classes)}

    return train_dataloader, test_dataloader, classes, class_dict
