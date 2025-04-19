import torch
from tqdm.notebook import tqdm

device = torch.device('mps')
# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, optimizer, criterion, vocab, epochs):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        batch_idx = 0
        for images, targets, _ in tqdm(train_loader):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            tgt_mask = model.transformer_decoder.generate_square_subsequent_mask(targets.size(1)-1).to(device)
            # tgt_mask = tgt_mask.repeat(images.size(0), 1, 1)
            tgt_padding_mask = (targets == vocab.char2idx['<pad>']).to(device)
            
            # Forward pass
            outputs, _ = model(images, targets[:, :-1], tgt_mask, tgt_padding_mask[:, :-1])
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: [{epoch+1}/{epochs}], Batch: [{batch_idx+1}/{len(train_loader)}], '
                        f'Loss: {loss.item():.4f}')
            batch_idx+=1
        print(f'Epoch: [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}')