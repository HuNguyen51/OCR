import numpy as np


def compute_accuracy(ground_truth, predictions, mode="full_sequence"):
    """
    Tính toán độ chính xác giữa ground truth và predictions
    
    Parameters:
    -----------
    ground_truth : list
        Danh sách các nhãn đúng
    predictions : list
        Danh sách các dự đoán
    mode : str, optional (default="full_sequence")
        Chế độ tính toán độ chính xác:
        - "per_char": Tính tỷ lệ ký tự đúng trên tổng số ký tự
        - "full_sequence": Chuỗi dự đoán phải chính xác hoàn toàn
        
    Returns:
    --------
    float
        Độ chính xác trung bình
    """
    if not ground_truth:
        return 1.0 if not predictions else 0.0
    
    if mode == "per_char":
        accuracies = []
        
        for label, pred in zip(ground_truth, predictions):
            if not label:  # Xử lý trường hợp nhãn rỗng
                accuracies.append(1.0 if not pred else 0.0)
                continue
                
            # Tính số ký tự đúng trong phạm vi chung
            min_len = min(len(label), len(pred))
            if min_len == 0:
                accuracies.append(0.0)
                continue
                
            # Đếm số ký tự trùng khớp
            correct_count = sum(l == p for l, p in zip(label[:min_len], pred[:min_len]))
            accuracies.append(correct_count / len(label))
            
        return np.mean(accuracies) if accuracies else 0.0
        
    elif mode == "full_sequence":
        correct_count = sum(pred == label for label, pred in zip(ground_truth, predictions))
        return correct_count / len(ground_truth)
    
    else:
        raise NotImplementedError(
            f"Chế độ tính độ chính xác '{mode}' chưa được triển khai"
        )