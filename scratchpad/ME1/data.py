import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MNISTData:
    """Owns the MNIST train/test datasets and hands out DataLoaders + random samples."""

    def __init__(self, root="data", batch_size_train=128, batch_size_test=256):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        )
        self.train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False)
        return train_loader, test_loader

    def sample_random_test_images(self, n=16, generator=None):
        """Returns (images, labels) for n randomly picked test-set examples."""
        indices = torch.randperm(len(self.test_dataset), generator=generator)[:n]
        images = torch.stack([self.test_dataset[i][0] for i in indices])
        labels = torch.tensor([self.test_dataset[i][1] for i in indices])
        return images, labels


if __name__ == "__main__":
    data = MNISTData()
    train_loader, test_loader = data.get_loaders()
    images, labels = next(iter(train_loader))
    print("train batch:", images.shape, images.dtype, "labels:", labels.shape, labels.dtype)
    print("train size:", len(data.train_dataset), "test size:", len(data.test_dataset))

    sample_images, sample_labels = data.sample_random_test_images(16)
    print("sample:", sample_images.shape, sample_labels.shape, sample_labels.tolist())
