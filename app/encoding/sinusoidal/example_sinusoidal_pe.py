import math
import torch
import torch.nn as nn

# 1. Sinusoidal Positional Encoder
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

# 2. Tiny Transformer Classifier / Next-Word Predictor
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])

# 3. Prepare Dataset
vocab = {"the": 0, "dog": 1, "bites": 2, "man": 3}
inv_vocab = {v: k for k, v in vocab.items()}

seq1 = [vocab[w] for w in ("the", "dog", "bites", "the")]
seq2 = [vocab[w] for w in ("the", "man", "bites", "the")]

X = torch.tensor((seq1, seq2))
y = torch.tensor((vocab["man"], vocab["dog"]))

# 4. Train the Model
torch.manual_seed(42)
model = MiniTransformer(vocab_size=len(vocab), d_model=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

for epoch in range(150):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

# 5. Test Predictions
model.eval()
with torch.no_grad():
    logits = model(X)
    predicted_ids = torch.argmax(logits, dim=-1)

for i, seq in enumerate(X):
    prompt = " ".join(inv_vocab[idx.item()] for idx in seq)
    pred_word = inv_vocab[predicted_ids[i].item()]
    print(f"Prompt: '{prompt}'  -->  Predicted next word: '{pred_word}'")
