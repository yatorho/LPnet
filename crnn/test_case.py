import torch
import torch.nn as nn

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'I', 'O']
ctc_table = provinces + ads
ctc_table.append('-')  # use '-' as blank

input = torch.randn(18, 4, 512)


class Mo(nn.Module):
    def __init__(self):
        super(Mo, self).__init__()
        self.rnn1 = nn.Sequential(
            nn.LSTM(512, 256, 3, bidirectional=True),
        ).cuda()
        self.rnn2 = nn.Sequential(
            nn.LSTM(512, 256, 3, bidirectional=True),
        ).cuda()
        self.rnn3 = nn.Sequential(
            nn.LSTM(512, 35, 3, bidirectional=True),
        ).cuda()
        self.ctc_loss = nn.CTCLoss(blank=len(ctc_table) - 1, reduction='mean', zero_infinity=True).cuda()

    def forward(self, input):
        output1, (h0, c0) = self.rnn1(input)
        output2, _ = self.rnn2(output1)
        output3, _ = self.rnn3(output2)
        output = nn.functional.log_softmax(output3, dim=2)
        return output

targest = torch.tensor([18, 45, 33, 37, 40, 49, 63, 4, 54, 51, 34, 53,
                        37, 38, 22, 56, 37, 38, 33, 39, 34, 46, 2, 41, 44, 37, 39, 35, 33, 40]).cuda()
m = Mo().cuda()
inputs = torch.randn(18, 4, 512).cuda()
optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = m(inputs)
    loss = m.ctc_loss(output, targest, torch.tensor([18, 18, 18, 18]).cuda(),  torch.tensor([7, 7, 8, 8]).cuda())
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss))

o = m(inputs)
print(o.shape)
# Get the max probability (greedy decoding) then decode the index to character
_, preds = o.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
preds_size = torch.IntTensor([preds.size(0)])
raw_pred = ''.join([ctc_table[x] for x in preds])
print(raw_pred)
sim_preds = raw_pred.strip('-').replace('-', '')
print(sim_preds)

# Compare with target and compute accuracy
targ = targest.cpu().numpy()
targ = targ[:preds_size]
raw_target = ''.join([ctc_table[x] for x in targ])
# print(raw_target)
sim_target = raw_target.strip('-').replace('-', '')
print(sim_target)

