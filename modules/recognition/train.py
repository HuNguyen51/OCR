import torch
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from modules.recognition.metrics import compute_accuracy

def train_model(model, train_loader, optimizer, criterion, device, vocab, epochs):
    model.train()
    model.to(device)
    num_iters = epochs*len(train_loader)

    scheduler = OneCycleLR(optimizer, total_steps=num_iters, max_lr=0.0003, pct_start=0.1)
    
    pad_idx = vocab.char2idx['<pad>']
    sos_idx = vocab.char2idx['<sos>']
    eos_idx = vocab.char2idx['<eos>']
    special_tokens = {pad_idx, sos_idx, eos_idx}

    for epoch in range(epochs):
        epoch_loss = 0
        acc_per_char = 0
        acc_full_sequence = 0
        len_loader = 0
        for images, targets, _ in tqdm(train_loader):
            # Move data to device
            batch_size = images.size(0)
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

            len_loader+=1
            if len_loader <= 1:
                # Dự đoán văn bản cho tính toán độ chính xác
                pred_texts = []
                target_texts = []
                
                # Tối ưu hoá bằng cách xử lý hàng loạt thay vì từng ảnh một
                batch_preds,_,_ = model.generate_text(images, vocab)
                pred_texts.extend(batch_preds)
                
                # Xử lý targets
                for i in range(batch_size):
                    target_text = ''.join([vocab.idx2char[idx.item()] 
                                        for idx in targets[i] if idx.item() not in special_tokens])
                    target_texts.append(target_text)

                # Tính toán độ chính xác ký tự
                for pred_text, target_text in zip(pred_texts, target_texts):
                    acc_per_char += compute_accuracy(pred_text, target_text, mode='per_char')/batch_size
                    acc_full_sequence += compute_accuracy(pred_text, target_text, mode='full_sequence')/batch_size

        avg_loss = epoch_loss / len_loader
        print(f'Epoch: [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        if acc_per_char == 0:
            continue
        avg_acc_per_char = acc_per_char #/ len_loader # chỉ tính trên 1 batch để tiết kiệm thời gian
        avg_acc_full_sequence = acc_full_sequence #/ len_loader
        print(f'Avg Accuracy per char: {avg_acc_per_char:.4f}, , Avg Accuracy full seq: {avg_acc_full_sequence:.4f}')
        
