import torch
import torch.nn.functional as F

from data import MNISTData
from model import EinopsCNN

NUM_EPOCHS = 5
LEARNING_RATE = 1e-3


def run_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += images.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total_examples = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += images.size(0)
    return total_correct / total_examples


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = MNISTData()
    train_loader, test_loader = data.get_loaders()

    model = EinopsCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device)
        print(f"epoch {epoch}/{NUM_EPOCHS} - train loss: {train_loss:.4f} - train acc: {train_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"final test accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    return model, test_acc


if __name__ == "__main__":
    main()
