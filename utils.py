import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch_lr_finder import LRFinder

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

def train(model, device, train_loader, optimizer, criterion, scheduler):
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
    scheduler.step()
    
    train_succeeded += GetCorrectPredCount(pred, target)
    train_processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*train_succeeded/train_processed:0.2f}')
  
  return train_succeeded, train_processed, train_loss


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    test_succeeded = 0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            batch_count += 1
            test_succeeded += GetCorrectPredCount(output, target)


    test_loss = test_loss / batch_count 
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_succeeded, len(test_loader.dataset),
        100. * test_succeeded / len(test_loader.dataset)))
    
    return test_succeeded, test_loss


def get_lr(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    end_lr=1,
    num_iter=100,
    step_mode="exp",
    start_lr=None,
    diverge_th=5,
):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        start_lr=start_lr,
        diverge_th=diverge_th,
    )
    _, max_lr = lr_finder.plot(log_lr=False, suggest_lr=True)

    # Reset the model and optimizer to initial state
    lr_finder.reset()

    return max_lr