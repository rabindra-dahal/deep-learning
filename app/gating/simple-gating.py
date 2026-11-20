import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class GLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        gate = torch.sigmoid(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class GeGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class ReGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        gate = F.relu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# Sample input: batch_size=2, seq_len=3, d_model=4
x = torch.tensor([
    [
        [1.0, -0.5, 2.0, -1.5],
        [0.2,  0.8, -0.4, 1.2],
        [-2.0, 1.5, -1.0, 0.5],
    ],
    [
        [0.5,  0.5,  0.5,  0.5],
        [-1.2, -0.8, 0.3, -0.2],
        [1.8, -2.2,  0.7,  1.1],
    ],
])

d_model = 4
hidden_dim = 8

modules = {
    "GLU": GLU(d_model, hidden_dim),
    "SwiGLU": SwiGLU(d_model, hidden_dim),
    "GeGLU": GeGLU(d_model, hidden_dim),
    "ReGLU": ReGLU(d_model, hidden_dim),
}

# Share identical initial weights across all 4 modules so differences come strictly from the gate activation
ref_gate = modules["GLU"].w_gate.weight.detach().clone()
ref_up = modules["GLU"].w_up.weight.detach().clone()
ref_down = modules["GLU"].w_down.weight.detach().clone()

for name, mod in modules.items():
    with torch.no_grad():
        mod.w_gate.weight.copy_(ref_gate)
        mod.w_up.weight.copy_(ref_up)
        mod.w_down.weight.copy_(ref_down)

    out = mod(x)
    print(f"=== {name} | output shape: {tuple(out.shape)} ===")
    print(out[0, 0].detach().round(decimals=4))  # Batch 0, Token 0 output vector
