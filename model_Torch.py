import torch.nn as nn
import torch

class CustomModel(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(CustomModel, self).__init__()

        self.fe1 = nn.Dropout(0.5)
        self.fe2 = nn.Linear(4096, 256)
        self.relu1 = nn.ReLU()

        self.se1 = nn.Embedding(vocab_size, 256 )
        self.se2 = nn.Dropout(0.5)
        self.se3 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        
        self.decoder1 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.decoder2 = nn.Linear(256, vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, captions):
        fe1_out = self.fe1(image_features)
        fe2_out = self.fe2(fe1_out)
        relu1_out = self.relu1(fe2_out)

        se1_out = self.se1(captions)
        se2_out = self.se2(se1_out)
        se3_out, (ht, ct) = self.se3(se2_out)

        # add_out = relu1_out + se3_out[:, -1, :]
        add_out = torch.cat( (torch.nn.functional.normalize(se3_out[:, -1, :]),
                                    torch.nn.functional.normalize(relu1_out)), dim=1)
        decoder1_out = self.decoder1(add_out)
        relu2_out = self.relu2(decoder1_out)
        decoder2_out = self.decoder2(relu2_out)
        
        outputs = self.softmax(decoder2_out)
        return outputs


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = (-(predictions+1e-5).log() * targets).sum(dim=1).mean()

        return loss

