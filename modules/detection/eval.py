from tqdm.notebook import tqdm
import torch
device = 'cpu'

# Training loop example (Pseudo-code)
def test_east(model, test_loader, criterion):
    """test loop for EAST model"""
    model.eval()
    model.to(device)
    epoch_loss=0
    batch_idx=0
    for images, score_maps_gt, geo_maps_gt, training_masks in tqdm(test_loader):
        # Move data to device
        images = images.to(device)
        score_maps_gt = score_maps_gt.to(device)
        geo_maps_gt = geo_maps_gt.to(device)
        training_masks = training_masks.to(device)
        
        # Forward pass
        with torch.no_grad():
            score_maps_pred, geo_maps_pred = model(images)
            
        # Calculate loss
        loss, dice_loss, geo_loss = criterion(
            score_maps_pred, geo_maps_pred,
            score_maps_gt, geo_maps_gt,
            training_masks
        )
        epoch_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch: [{batch_idx+1}/{len(test_loader)}], '
                    f'Loss: {loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, Geo Loss: {geo_loss.item():.4f}')
        batch_idx+=1
    print(f'Average Loss: {epoch_loss/len(test_loader):.4f}')
    return epoch_loss/len(test_loader)

# Training loop example (Pseudo-code)
def test_east(model, test_loader, criterion):
    """test loop for EAST model"""
    model.eval()
    model.to(device)
    epoch_loss=0
    batch_idx=0
    for images, score_maps_gt, training_masks in tqdm(test_loader):
        # Move data to device
        images = images.to(device)
        score_maps_gt = score_maps_gt.to(device)
        training_masks = training_masks.to(device)
        
        # Forward pass
        with torch.no_grad():
            score_maps_pred = model(images)
            
        # Calculate loss
        loss = criterion(
            score_maps_pred,
            score_maps_gt
        )
        epoch_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch: [{batch_idx+1}/{len(test_loader)}], '
                    f'Loss: {loss.item():.4f}')
        batch_idx+=1
    print(f'Average Loss: {epoch_loss/len(test_loader):.4f}')
    return epoch_loss/len(test_loader)