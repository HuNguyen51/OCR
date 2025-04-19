import torch
from tqdm.notebook import tqdm

# Hàm đánh giá mô hình
def evaluate(model, val_loader, criterion, device, vocabulary):
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Dự đoán văn bản
            pred_texts = []
            for i in range(images.size(0)):
                pred_text, _, _ = model.generate_text(images[i:i+1], vocabulary)
                pred_texts.append(pred_text)
            
            # Tính toán Character Error Rate (CER)
            for i in range(len(pred_texts)):
                target_text = ''.join([vocabulary.idx2char[idx.item()] for idx in targets[i] 
                                       if idx.item() not in [vocabulary.char2idx['<sos>'], 
                                                            vocabulary.char2idx['<eos>'], 
                                                            vocabulary.char2idx['<pad>']]])
                
                # So sánh từng ký tự
                min_len = min(len(pred_texts[i]), len(target_text))
                for j in range(min_len):
                    if pred_texts[i][j] == target_text[j]:
                        correct_chars += 1
                total_chars += max(len(pred_texts[i]), len(target_text))
            
            # Forward pass
            tgt_mask = model.transformer_decoder.generate_square_subsequent_mask(targets.size(1)-1).to(device)
            tgt_padding_mask = (targets == vocabulary.char2idx['<pad>']).to(device)
            
            outputs, _ = model(images, targets[:, :-1], tgt_mask, tgt_padding_mask[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    character_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    print(f'Evaluation - Avg Loss: {avg_loss:.4f}, Character Accuracy: {character_accuracy:.4f}')
    return avg_loss, character_accuracy

# Hàm dự đoán trên tập test
def predict(model, test_loader, vocabulary, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            for i in range(images.size(0)):
                pred_text, img_features = model.generate_text(images[i:i+1], vocabulary)
                results.append((images[i], pred_text, img_features))
    
    return results
