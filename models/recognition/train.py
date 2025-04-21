import torch
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from models.recognition.metrics import compute_accuracy
from torch.nn.functional import softmax
device = torch.device('mps')
# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, optimizer, criterion, vocab, epochs):
    model.train()
    model.to(device)
    num_show = len(train_loader)//5
    num_iters = epochs*len(train_loader)
    scheduler = OneCycleLR(optimizer, total_steps=num_iters, max_lr=0.0003, pct_start=0.1)
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_idx = 0
        for images, targets, _ in tqdm(train_loader):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)

            tgt_in = targets[:,:-1]
            tgt_out = targets[:,1:]
            
            tgt_mask = model.transformer_decoder.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            tgt_padding_mask = (tgt_in == vocab.char2idx['<pad>']).to(device)
            
            # Forward pass
            outputs, _ = model(images, tgt_in, tgt_mask, tgt_padding_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_out = tgt_out.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, tgt_out)

            # Backward pass and optimization
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            # actual_sent = vocab.batch_decode(tgt_out)
            # pred_sent = vocab.batch_decode(model.generate_text(images, vocab, len(tgt_out[0])))
            # acc_full_seq, acc_per_char = precision(actual_sent, pred_sent)

            if (batch_idx + 1) % num_show == 0:
                print(f'Epoch: [{epoch+1}/{epochs}], Batch: [{batch_idx+1}/{len(train_loader)}], '
                        f'Loss: {loss.item():.4f}')
            batch_idx+=1
        print(f'Epoch: [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}')

def precision(actual_sents, pred_sents):

    acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode="full_sequence")
    acc_per_char = compute_accuracy(actual_sents, pred_sents, mode="per_char")

    return acc_full_seq, acc_per_char