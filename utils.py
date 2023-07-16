import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def print_featuremaps_hook(self, input, output):
      # Detach one output feature map (one channel)
      for i in range(output.shape[1]):
          feature_map = output[0, i].detach().cpu().numpy()
          
          # Plot the feature map
          plt.figure(figsize=(3, 3))
          plt.imshow(feature_map, cmap='gray')
          plt.show()

def show_batch_images(plt, dataloader, count=12, row = 3, col = 4):
    images, labels = next(iter(dataloader))
    for i in range(count):
        plt.subplot(row, col, i+1)
        plt.tight_layout()
        plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.title(labels[i].item())
        plt.xticks([])
        plt.yticks([])

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  train_succeeded = 0
  train_processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    train_succeeded += GetCorrectPredCount(pred, target)
    train_processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*train_succeeded/train_processed:0.2f}')
  
  return train_succeeded, train_processed, train_loss


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    test_succeeded = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            test_succeeded += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_succeeded, len(test_loader.dataset),
        100. * test_succeeded / len(test_loader.dataset)))
    
    return test_succeeded, test_loss