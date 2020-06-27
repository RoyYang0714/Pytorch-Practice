import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data

train_x = [torch.tensor([1, 1, 1, 1, 1, 1, 1]),
           torch.tensor([2, 2, 2, 2, 2, 2]),
           torch.tensor([3, 3, 3, 3, 3]),
           torch.tensor([4, 4, 4, 4]),
           torch.tensor([5, 5, 5]),
           torch.tensor([6, 6]),
           torch.tensor([7])]

class MyData(data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]

def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data.unsqueeze(-1), data_length

if __name__=='__main__':
    data = MyData(train_x)
    data_loader = DataLoader(data, batch_size=3, shuffle=True, 
                             collate_fn=collate_fn)
    batch_x, batch_x_len = iter(data_loader).next()
    batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, 
                                                  batch_x_len, batch_first=True).float()
    
    net = nn.LSTM(1, 10, 2, batch_first=True)
    h0 = torch.rand(2, 3, 10)
    c0 = torch.rand(2, 3, 10)
    out, (h1, c1) = net(batch_x_pack, (h0, c0))

    print(out.data.shape)

    out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)

    print(out_pad.shape)
