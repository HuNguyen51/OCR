# Tạo từ điển ký tự
class Vocabulary:
    __instance=None
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Vocabulary, cls).__new__(cls, *args, **kwargs)
        return cls.__instance
    
    def __init__(self):
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.num_chars = 4  # Số ký tự hiện tại trong từ điển
        
        # Thêm các ký tự chữ cái, số và dấu câu phổ biến
        general_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        numbers = '0123456789'
        special_chars = ' ,.!?-_\'"|%$@#^&*=+:;()[]{}'
        vietnamese_chars = 'ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ'
        
        for c in general_chars + numbers + special_chars + vietnamese_chars:
            self.add_char(c)

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.num_chars
            self.idx2char[self.num_chars] = char
            self.num_chars += 1
    
    def __len__(self):
        return self.num_chars