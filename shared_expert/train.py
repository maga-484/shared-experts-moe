import torch
import torch.optim as optim
from .model import SharedExpertMoE
from .generator import generate_silo_data

def train_initial(num_silos=10, epochs=100, lr=0.001, verbose=True):
    model = SharedExpertMoE(num_specialized=num_silos)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    silos_data = []
    for i in range(num_silos):
        variation = 0.05 + (i / num_silos) * 0.45
        X, y = generate_silo_data(seed=1000+i, n_samples=500, variation=variation)
        silos_data.append((X, y))
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for silo_id, (X, y) in enumerate(silos_data):
            optimizer.zero_grad()
            pred = model(X, silo_id)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_silos
        losses.append(avg_loss)
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")
    return model, losses, silos_data
