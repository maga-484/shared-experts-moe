import torch
import torch.optim as optim
from .model import SharedExpertMoE

def update_for_new_domain(model, silo_id, X_new, y_new, X_old, y_old,
                          steps=100, lr=0.001, replay_weight=0.5):
    model.freeze_shared()
    optimizer = optim.Adam(model.heads[silo_id].parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        pred_old = model(X_old, silo_id)
        pred_new = model(X_new, silo_id)
        loss = (1 - replay_weight) * loss_fn(pred_new, y_new) + replay_weight * loss_fn(pred_old, y_old)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    model.unfreeze_shared()
    return losses
