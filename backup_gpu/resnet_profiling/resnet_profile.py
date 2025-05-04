import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import nvtx
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy dataset
def get_dummy_data(batch_size=32, num_classes=1000):
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return inputs, labels

# Model
model = models.resnet18(pretrained=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training step
@nvtx.annotate("Training", color="blue")
def train_step():
    model.train()
    inputs, labels = get_dummy_data()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Inference step
@nvtx.annotate("Inference", color="green")
def inference_step():
    model.eval()
    inputs, _ = get_dummy_data()
    with torch.no_grad():
        _ = model(inputs)

# Main loop
if __name__ == "__main__":
    # Warm-up
    for _ in range(2):
        train_step()
        inference_step()

    torch.cuda.profiler.start()

    for epoch in range(5):
        train_step()
        inference_step()

    torch.cuda.profiler.stop()

    torch.cuda.synchronize()
    print("Profiling run complete.")

