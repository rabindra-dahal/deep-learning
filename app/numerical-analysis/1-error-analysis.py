import torch

# Force double precision (64-bit) for comparison, and single precision (32-bit)
print("=== 1. THE 0.1 REPEATING BINARY TRAP ===")
a = torch.tensor([0.1], dtype=torch.float32)
b = torch.tensor([0.1], dtype=torch.float64)

# Print with high decimal precision to reveal hidden rounding
print(f"32-bit Float 0.1 looks like: {a.item():.20f}")
print(f"64-bit Float 0.1 looks like: {b.item():.20f}")
print(f"Is 0.1 + 0.2 exactly 0.3? {(a + 0.2).item() == 0.3}\n")


print("=== 2. CATASTROPHIC CANCELLATION ===")
# Two very close numbers
x = torch.tensor([1.0000001], dtype=torch.float32)
y = torch.tensor([1.0000000], dtype=torch.float32)

# The mathematical answer should be exactly 0.0000001 (1e-7)
difference = x - y
print(f"True mathematical difference: 0.0000001000...")
print(f"Actual computer calculation   : {difference.item():.10f}")
