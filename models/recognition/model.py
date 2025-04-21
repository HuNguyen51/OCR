import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import math
import random

# CNN Backbone (ResNet18 với thay đổi)
class CNNBackbone(nn.Module):
    def __init__(self, output_dim=512):
        super(CNNBackbone, self).__init__()
        # Sử dụng ResNet (VGG16) làm backbone hoặc xây dựng CNN từ đầu
        # Ví dụ sử dụng ResNet (VGG16) đã pre-trained nhưng chỉ lấy phần feature extraction
        cnn = models.vgg16_bn(weights='DEFAULT')
        # Bỏ lớp fully connected cuối cùng
        self.backbone = nn.Sequential(*list(cnn.children())[:-2])
        
        # Projection layer để biến đổi feature map thành dạng phù hợp với transformer
        self.projection = nn.Conv2d(512, output_dim, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, 3, height, width]
        features = self.backbone(x)
        # features shape: [batch_size, 512, h', w']
        features = self.projection(features)
        # features shape: [batch_size, output_dim, h', w']
        
        # Reshape để phù hợp với transformer
        batch_size, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).contiguous()
        # features shape: [batch_size, h', w', output_dim]
        features = features.view(batch_size, h * w, c)
        # features shape: [batch_size, h'*w', output_dim]
        
        return features

# Tự triển khai Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return self.fc2(self.relu(self.fc1(x)))

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Tạo positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Đăng ký pe như là buffer (không phải là parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Tự triển khai DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True) 
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention với memory (CNN features)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed forward
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, key_padding_mask=None, return_attention=False):
        # x shape: [batch_size, tgt_len, d_model]
        # memory shape: [batch_size, src_len, d_model]
        
        # Masked Self-attention
        residual = x

        x, self_attn = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(residual + x)
        
        # Cross-attention
        residual = x
        x, cross_attn = self.cross_attn(x, memory, memory)
        x = self.dropout2(x)
        x = self.norm2(residual + x)
        
        # Feed forward
        residual = x
        x = self.ff(x)
        x = self.dropout3(x)
        x = self.norm3(residual + x)
        
        if return_attention:
            return x, self_attn, cross_attn
        return x

# Tự triển khai TransformerDecoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Stack các decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        # Tạo mask để ngăn chặn tự attention vào các vị trí tương lai
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, return_attention=False):
        # tgt shape: [batch_size, tgt_len]
        # memory shape: [batch_size, src_len, d_model]
        
        # Embedding và positional encoding
        tgt_embeddings = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        
        # Process through decoder layers
        x = tgt_embeddings
        attentions = []
        
        for layer in self.layers:
            if return_attention:
                x, self_attn, cross_attn = layer(x, memory, tgt_mask, tgt_key_padding_mask, return_attention=True)
                attentions.append((self_attn, cross_attn))
            else:
                x = layer(x, memory, tgt_mask, tgt_key_padding_mask)
        
        x = self.norm(x)
        
        # Project back to vocabulary
        output = self.output_projection(x)
        
        if return_attention:
            return output, attentions
        return output

# OCR Model kết hợp CNN và Transformer Decoder được triển khai từ đầu
class OCRModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_decoder_layers=6, dropout=0.1):
        super(OCRModel, self).__init__()
        
        # CNN để trích xuất đặc trưng
        self.cnn_backbone = CNNBackbone(output_dim=d_model)
        
        # Transformer Decoder tự triển khai
        self.transformer_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
        self.device = torch.device('mps')
    
    def forward(self, img, tgt, tgt_mask=None, tgt_padding_mask=None, return_attention=False):
        # img shape: [batch_size, channels, height, width]
        # tgt shape: [batch_size, seq_len]
        
        # Trích xuất đặc trưng từ ảnh bằng CNN
        img_features = self.cnn_backbone(img)
        # img_features shape: [batch_size, h*w, d_model]
        
        # Sử dụng Transformer Decoder để sinh văn bản
        if return_attention:
            output, attentions = self.transformer_decoder(
                tgt, img_features, tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_padding_mask, 
                return_attention=True
            )
            return output, img_features, attentions
        
        else:
            output = self.transformer_decoder(
                tgt, img_features, tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_padding_mask
            )
            return output, img_features
    
    def generate_text(self, img, vocabulary, max_length=100):
        # Dùng để dự đoán văn bản từ ảnh
        self.eval()
        
        with torch.no_grad():
            # Trích xuất đặc trưng ảnh
            img_features = self.cnn_backbone(img)
            # img_features: [batch_size, h*w, d_model]
            
            batch_size = img.size(0)
            
            # Khởi tạo token <sos> (start of sequence)
            ys = torch.ones(batch_size, 1).fill_(vocabulary.char2idx['<sos>']).long().to(self.device)
            
            # Lưu lại các cross-attention weights cho visualization
            all_cross_attentions = []
            
            for i in range(max_length - 1):
                # Tạo attention mask để không nhìn phía trước
                tgt_mask = self.transformer_decoder.generate_square_subsequent_mask(ys.size(1)).to(self.device)
                # Decode với attention
                out, _, attentions = self(img, ys, tgt_mask, return_attention=True)
                # attentions: [num_layers][attention][batch_size]
                # out: [batch_size, seq_len, vocab_size]
                # Lấy cross-attention weights từ lớp decoder cuối cùng cho token cuối cùng
                last_layer_cross_attn = attentions[-1][1]  # [batch_size, num_heads, seq_len, src_len]
                # Lấy weights của token cuối cùng
                token_cross_attn = last_layer_cross_attn[:, -1, :]  # [batch_size, src_len]
                # Tính trung bình qua tất cả heads
                all_cross_attentions.append(token_cross_attn)

                # Lấy dự đoán tốt nhất cho token cuối cùng
                prob = out[:, -1]
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(1)
                
                # Thêm token dự đoán vào chuỗi đầu ra
                ys = torch.cat([ys, next_word], dim=1)

                # Dừng nếu token là <eos>
                if next_word[0].item() == vocabulary.char2idx['<eos>']:
                    break
            
            # Chuyển đổi indices thành ký tự
            texts = []
            for predict_seq in ys.cpu().numpy():
                text = vocabulary.decode(predict_seq)
                texts.append(text)

            # Stack attention weights vào tensor
            if all_cross_attentions:
                # all_cross_attentions: max_len, batch_size, attention_weights
                all_cross_attentions = torch.stack(all_cross_attentions)
            else:
                all_cross_attentions = torch.tensor([])
            
            return texts, img_features, all_cross_attentions
    
    def visualize_attention(self, img, text, text_generated, img_features, attention_weights=None):
        if text != '':
            print("actual: ",text)
        print("generated: ",text_generated)
        # Hiển thị hình ảnh và visualize attention weights
        self.eval()
        
        # Reshape lại feature map để có thể hiển thị
        # feature_size = int(math.sqrt(img_features.size(1)))
        
        plt.figure(figsize=(15, 10))
        
        # Hiển thị ảnh gốc
        plt.subplot(1, 2, 1)
        # Chuyển tensor sang numpy để hiển thị
        img_np = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
        # Unnormalize ảnh
        # img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        
        # Hiển thị heatmap attention
        plt.subplot(1, 2, 2)
        
        if attention_weights is not None and len(attention_weights) > 0:
            # Lấy attention weights trung bình nếu có nhiều ký tự
            avg_attention = attention_weights.mean(0)
            
            # Reshape lại thành grid để dễ visualize
            # attention_map = avg_attention.reshape(feature_size, feature_size).cpu().numpy()
            
            # Hiển thị attention map
            plt.imshow(avg_attention.cpu(), cmap='hot')
            plt.title(f"Attention Heatmap for text:")
            plt.colorbar()
        else:
            plt.text(0.5, 0.5, "No attention weights available", 
                     horizontalalignment='center',
                     verticalalignment='center')
            plt.title("Attention Heatmap")
        
        plt.tight_layout()
        plt.show()