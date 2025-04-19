# Check that MPS is available
import torch

device = torch.device("mps")
# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm.notebook import tqdm

def train_east(model, train_loader, optimizer, criterion, epochs):
    """Training loop for EAST model"""
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_idx = 0
        for images, score_maps_gt, geo_maps_gt, training_masks in tqdm(train_loader):
            # Move data to device
            images = images.to(device)
            score_maps_gt = score_maps_gt.to(device)
            geo_maps_gt = geo_maps_gt.to(device)
            training_masks = training_masks.to(device)
            
            # Forward pass
            score_maps_pred, geo_maps_pred = model(images)
            
            # Calculate loss
            loss, dice_loss, geo_loss = criterion(
                score_maps_pred, geo_maps_pred,
                score_maps_gt, geo_maps_gt,
                training_masks
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: [{epoch+1}/{epochs}], Batch: [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, Geo Loss: {geo_loss.item():.4f}')
            batch_idx+=1
        print(f'Epoch: [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}')

def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        batch_idx = 0
        for images, score_maps_gt, training_masks in tqdm(train_loader):
            # Move data to device
            images = images.to(device)
            score_maps_gt = score_maps_gt.to(device)
            training_masks = training_masks.to(device)
            
            # Forward pass
            score_maps_pred = model(images)
            
            score_maps_pred=score_maps_pred.squeeze(1)

            # Calculate loss
            loss = criterion(
                score_maps_pred,
                score_maps_gt
            )
            loss*=10
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 25 == 0:
                print(f'Epoch: [{epoch+1}/{epochs}], Batch: [{batch_idx+1}/{len(train_loader)}], '
                        f'Loss: {loss.item():.4f}')
            batch_idx+=1
        print(f'Epoch: [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}')