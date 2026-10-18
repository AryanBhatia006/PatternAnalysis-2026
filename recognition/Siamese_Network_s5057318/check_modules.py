"""Smoke test for BaselineCNN: output shape, parameter count and gradient flow."""
import torch
import torch.nn as nn

from modules import BaselineCNN

torch.manual_seed(0)
model = BaselineCNN()
x = torch.randn(4, 3, 224, 224)
logits = model(x)
print("output shape:", tuple(logits.shape))
assert logits.shape == (4, 1), "expected one logit per image"

trainable = [p for p in model.parameters() if p.requires_grad]
print("trainable parameters:", sum(p.numel() for p in trainable))

labels = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
loss = nn.BCEWithLogitsLoss()(logits, labels)
loss.backward()
print("loss:", round(loss.item(), 4))
with_grad = sum(p.grad is not None for p in trainable)
print(f"parameters with a gradient: {with_grad} / {len(trainable)}")
assert with_grad == len(trainable), "some parameters received no gradient"
