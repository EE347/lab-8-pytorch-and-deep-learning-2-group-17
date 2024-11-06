import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Hyperparameters
    learning_rate = 0.001
    batch_size = 8
    patience = 10  # Early stopping patience

    # Create the datasets and dataloaders with modified batch size
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # CSV file to store results
    with open('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/results_final.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Learning Rate', 'Batch Size', 'Train Loss', 'Test Loss', 'Test Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
        best_train_loss = 1e9
        train_losses = []
        test_losses = []
        test_accuracies = []

        # Early stopping variables
        best_test_loss = float('inf')
        early_stop_counter = 0

        print("\nTraining with CrossEntropyLoss...\n")
        for epoch in range(1, 3):
            t = time.time_ns()
            model.train()
            train_loss = 0

            for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
                images = images.reshape(-1, 3, 64, 64).to(device)
                labels = labels.to(device)

                # Apply random horizontal flip and rotation
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
            all_preds = []
            all_labels = []

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

                # Collect all predictions and actual labels for the confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            test_accuracy = correct / total
            print(f"Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s")

            train_losses.append(train_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))
            test_accuracies.append(test_accuracy)

            writer.writerow({
                'Epoch': epoch,
                'Learning Rate': learning_rate,
                'Batch Size': batch_size,
                'Train Loss': train_loss / len(trainloader),
                'Test Loss': test_loss / len(testloader),
                'Test Accuracy': test_accuracy
            })

            # Learning rate scheduler step
            scheduler.step(test_loss)

            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/best_model_final.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Generate and save the confusion matrix after each epoch
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/confusion_matrix_final.png')
            plt.clf()

        # Save final model and training metrics
        torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/final_model_final.pth')
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/final_loss_plot.png')
        plt.clf()

        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/final_accuracy_plot.png')
        plt.clf()