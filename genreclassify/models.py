import torch
import torch.nn as nn
import torch



device = "cuda" if torch.cuda.is_available() else "cpu"

class LstmCell(nn.Module):

    def __init__(self, h1_in, h2_out, output_dim):
        super().__init__()

        self.rnn = nn.LSTM(
            h1_in,
            h2_out,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(h2_out * 2, output_dim)


    def forward(self, x):
        # x -> [batch, seq, dim]

        _, (hidden, _) = self.rnn(x)
        # hidden -> [n_layers * n_directions, seq, hidden_dim]


        # cat two hidden dimension
        hidden = torch.cat([hidden[-2,:, :], hidden[-1, :, :]], dim=1)
        prediction = self.fc(hidden)
        return prediction
