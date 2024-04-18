import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FFNN(nn.Module):
    def __init__(self, num_agents_per_scenario, num_states_steps, num_future_steps):
        super(FFNN, self).__init__()
        self.num_agents_per_scenario = num_agents_per_scenario
        self.num_future_steps = num_future_steps
        self.dense1 = nn.Linear(num_states_steps * 7, 128)
        self.dense2 = nn.Linear(128, 256)
        self.dense3 = nn.Linear(256, 128)
        self.regressor = nn.Linear(128, num_future_steps * 7)
        self.conf_regressor = nn.Linear(128, 1) 

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the last two dimensions
        x = x.view(-1, x.size(-1))  # Flatten to [batch_size * num_agents, features]

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))

        pred_trajectory = self.regressor(x).view(4, self.num_agents_per_scenario, self.num_future_steps, 7)
        confidence_score = torch.sigmoid(self.conf_regressor(x)).view(-1, self.num_agents_per_scenario, 1)
        return pred_trajectory, confidence_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):
    def __init__(self, num_agents_per_scenario, num_states_steps, num_future_steps, d_model=128, nhead=16, num_encoder_layers=3):
        super(Transformer, self).__init__()
        self.num_agents_per_scenario = num_agents_per_scenario
        self.num_future_steps = num_future_steps
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder_layer.self_attn.batch_first = True
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_fc = nn.Linear(num_states_steps * 7, d_model) 
        self.regressor = nn.Linear(d_model, num_future_steps * 7)
        self.conf_regressor = nn.Linear(d_model, 1) 

    def forward(self, x):
        # x is of shape [batch_size, num_agents, num_states_steps, state_space=7]
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the last two dimensions
        x = self.input_fc(x)  # Match dimension to d_model for transformer
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, features]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to [batch_size, num_agents, d_model]
        
        # Splitting the tensor to separate predictions and confidence computations
        pred_trajectory = self.regressor(x).view(-1, self.num_agents_per_scenario, self.num_future_steps, 7)
        confidence_score = torch.sigmoid(self.conf_regressor(x)).view(-1, self.num_agents_per_scenario, 1)
        return pred_trajectory, confidence_score
