# Tạo từ điển ký tự
class Vocabulary:
    _instance=None
    def __new__(cls, maxlen=None):
        if cls._instance is None:
            cls._instance = super(Vocabulary, cls).__new__(cls)
            cls._instance._initialize(maxlen)
        return cls._instance
    
    def _initialize(self, maxlen=None):
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.num_chars = 4  # Số ký tự hiện tại trong từ điển
        
        # Thêm các ký tự chữ cái, số và dấu câu phổ biến
        general_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        numbers = '0123456789'
        special_chars = '!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ '
        vietnamese_chars = 'ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ'

        for c in general_chars + numbers + special_chars + vietnamese_chars:
            self.add_char(c)
        self.maxlen = maxlen

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.num_chars
            self.idx2char[self.num_chars] = char
            self.num_chars += 1
    
    def __len__(self):
        return self.num_chars
    
    @staticmethod
    def padding(text_ids, maxlen):
        if len(text_ids) < maxlen:
            text_ids += [0]*(maxlen-len(text_ids))
        else:
            text_ids = text_ids[:maxlen]
        return text_ids
    
    def encode(self, text):
        target = [self.char2idx['<sos>']]
        for char in text:
            if char in self.char2idx:
                target.append(self.char2idx[char])
            else:
                target.append(self.char2idx['<unk>'])
        target.append(self.char2idx['<eos>'])
        if self.maxlen:
            target = self.padding(target, self.maxlen)
        return target
    
    def decode(self, text_ids):
        text = []
        for idx in text_ids:
            if idx == self.char2idx['<eos>']:
                break
            if idx == self.char2idx['<sos>']:
                continue
            text.append(self.idx2char[idx])
        return ''.join(text)