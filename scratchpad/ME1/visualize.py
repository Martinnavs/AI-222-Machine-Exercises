import matplotlib.pyplot as plt
import torch

from data import MNISTData
from model import EinopsCNN

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def unnormalize(image):
    """Undo the Normalize(mean, std) transform for display."""
    return image * MNIST_STD + MNIST_MEAN


@torch.no_grad()
def plot_predictions_grid(model, images, labels, device):
    model.eval()
    logits = model(images.to(device))
    predictions = logits.argmax(dim=1).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, image, gt, pred in zip(axes.flat, images, labels, predictions):
        ax.imshow(unnormalize(image.squeeze(0)), cmap="gray")
        is_correct = gt.item() == pred.item()
        ax.set_title(f"gt={gt.item()} pred={pred.item()}", color="green" if is_correct else "red")
        ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EinopsCNN().to(device)
    model.load_state_dict(torch.load("checkpoint.pt", map_location=device, weights_only=True))

    data = MNISTData()
    images, labels = data.sample_random_test_images(16)

    fig = plot_predictions_grid(model, images, labels, device)
    fig.savefig("sample_grid.png")
    print("saved sample_grid.png")
