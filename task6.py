import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Hyperparameters
    learning_rate = 0.001  # Try adjusting this value (e.g., 0.0001, 0.005, 0.01)
    batch_size = 8  # Try adjusting this value (e.g., 4, 8, 16, 32)

    # Create the datasets and dataloaders with modified batch size
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Create the model and optimizer with modified learning rate
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_train_loss = 1e9
    train_losses = []
    test_losses = []

    for epoch in range(1, 100):
        t = time.time_ns()
        model.train()
        train_loss = 0

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Apply random horizontal flip and rotation as in Task 4
            flip_prob = 0.5
            flip_mask = torch.rand(images.size(0)) < flip_prob
            if flip_mask.any():
                images[flip_mask] = torch.flip(images[flip_mask], dims=[3])

            rotation_degrees = 10
            rotated_images = []
            for img in images:
                angle = (torch.rand(1).item() * 2 - 1) * rotation_degrees
                center = (img.shape[2] // 2, img.shape[1] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img.cpu().numpy().transpose(1, 2, 0), rotation_matrix, (img.shape[2], img.shape[1]))
                rotated_images.append(torch.from_numpy(rotated_img).permute(2, 0, 1).to(device))
            images = torch.stack(rotated_images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            flip_prob = 0.5
            flip_mask = torch.rand(images.size(0)) < flip_prob
            if flip_mask.any():
                images[flip_mask] = torch.flip(images[flip_mask], dims=[3])

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/best_model.pth')

        torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/current_model.pth')
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/task6_loss_plot.png')