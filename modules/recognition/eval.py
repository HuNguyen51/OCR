import torch
from tqdm.notebook import tqdm
from modules.recognition.metrics import compute_accuracy

def evaluate(model, val_loader, criterion, device, vocabulary):
    """
    Đánh giá mô hình trên tập validation
    
    Args:
        model: Mô hình cần đánh giá
        val_loader: DataLoader cho tập validation
        criterion: Hàm loss
        device: Thiết bị tính toán (CPU/GPU)
        vocabulary: Từ điển ánh xạ ký tự
        
    Returns:
        avg_loss: Loss trung bình
        character_accuracy: Độ chính xác ký tự
    """
    model.eval()
    model.to(device)
    total_loss = 0
    acc_per_char = 0
    acc_full_sequence = 0
    
    # Xác định trước các token đặc biệt để tránh tra cứu lặp lại
    pad_idx = vocabulary.char2idx['<pad>']
    sos_idx = vocabulary.char2idx['<sos>']
    eos_idx = vocabulary.char2idx['<eos>']
    special_tokens = {pad_idx, sos_idx, eos_idx}
    
    len_loader = 0
    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader):
            len_loader +=1
            batch_size = images.size(0)
            images = images.to(device)
            targets = targets.to(device)

            tgt_in = targets[:,:-1]
            tgt_out = targets[:,1:]
            
            # Forward pass cho loss calculation
            tgt_mask = model.transformer_decoder.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            tgt_padding_mask = (tgt_in == pad_idx).to(device)
            outputs, _ = model(images, tgt_in, tgt_mask, tgt_padding_mask)

            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_out = tgt_out.reshape(-1)

            loss = criterion(outputs, tgt_out)
            total_loss += loss.item()
            
            # Dự đoán văn bản cho tính toán độ chính xác
            pred_texts = []
            target_texts = []
            
            # Tối ưu hoá bằng cách xử lý hàng loạt thay vì từng ảnh một
            batch_preds,_,_ = model.generate_text(images, vocabulary)
            pred_texts.extend(batch_preds)
            
            # Xử lý targets
            for i in range(batch_size):
                target_text = ''.join([vocabulary.idx2char[idx.item()] 
                                     for idx in targets[i] if idx.item() not in special_tokens])
                target_texts.append(target_text)

            # Tính toán độ chính xác ký tự
            for pred_text, target_text in zip(pred_texts, target_texts):
                acc_per_char += compute_accuracy(pred_text, target_text, mode='per_char')/batch_size
                acc_full_sequence += compute_accuracy(pred_text, target_text, mode='full_sequence')/batch_size
            
    
    avg_loss = total_loss / len_loader
    avg_acc_per_char = acc_per_char / len_loader
    avg_acc_full_sequence = acc_full_sequence / len_loader
    print(f'Evaluation - Avg Loss: {avg_loss:.4f}, Avg Accuracy per char: {avg_acc_per_char:.4f}, , Avg Accuracy full seq: {avg_acc_full_sequence:.4f}')
    
    return avg_loss, avg_acc_per_char, avg_acc_full_sequence