import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary

def get_correct_predict_count(pPrediction, pLabels, pred_by_labels, label_count):
    
    for i in range(0, 10):
        label_items = pLabels.eq(torch.from_numpy(np.full((1, len(pLabels)), i)))
        lab_count = label_items.sum().item()
        pred_items = pPrediction.argmax(dim=1).eq(pLabels)
        correct_label = pred_items.logical_and(label_items).sum().item()
        if label_count == 0:
            continue
        pred_by_labels[i] += correct_label
        label_count[i] += lab_count

    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def plot_data(data_loader):
    batch_iter = iter(data_loader);
    batch_data, batch_label = next(batch_iter) 
    print("Batch Size {}".format(batch_data.shape));

    fig = plt.figure()

    for i in range(128):
        plt.subplot(13,10,i+1)
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        # plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def is_cuda_available():
   return torch.cuda.is_available()

def get_dst_device():
    return torch.device("cuda" if is_cuda_available() else "cpu")

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def setup_train_loader(destination):
    batch_size = 128
    kwargs = {
        'batch_size': batch_size, 
        'shuffle': True, 
        'num_workers': 2, 
        'pin_memory': True
    }
    train_data = datasets.MNIST(destination, train=True, download=True, transform=get_train_transforms())
    return torch.utils.data.DataLoader(train_data, **kwargs)

def setup_test_loader(destination):
    batch_size = 128
    kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True
    }
    test_data = datasets.MNIST(destination, train=False, download=True, transform=get_test_transforms())
    return torch.utils.data.DataLoader(test_data, **kwargs)


def train_model(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0
  pred_by_labels = [0 for i in range(10)];
  label_count = [0 for i in range(10)];
  label_pred = [0 for i in range(10)];

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += get_correct_predict_count(pred,
                                         target, pred_by_labels, label_count)
    processed += len(data)

    pbar.set_description(
        desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_accuracy = 100 * correct / processed

    train_loss /= len(train_loader)

  for i in range(10):
    label_pred[i] = 100 * (pred_by_labels[i] / label_count[i])

  print(label_pred)

  return [ train_accuracy, train_loss ]


def test_model(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0
    pred_by_labels = [0 for i in range(10)];
    label_count = [0 for i in range(10)];
    label_pred = [0 for i in range(10)]; 

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += get_correct_predict_count(output, target, pred_by_labels, label_count)


    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    for i in range(10):
        label_pred[i] = 100 * (pred_by_labels[i] / label_count[i])

    print(label_pred)
    return [test_accuracy, test_loss]

def plot_results(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def print_model_summary(model):
    summary(model, input_size=(1, 28, 28)) 