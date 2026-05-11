import torch
import sys
sys.path.append('.')
from shared_expert.train import train_initial
from .update import update_for_new_domain
from .generator import generate_silo_data

def test_shift():
    print("="*60)
    print("Prueba de cambio numérico gradual con Shared Expert MoE")
    print("="*60)
    
    print("\n[1] Entrenando modelo inicial con 10 silos...")
    model, losses, silos_data = train_initial(num_silos=10, epochs=100, lr=0.001, verbose=False)
    print(f"    Pérdida final promedio: {losses[-1]:.6f}")
    
    silo_id = 0
    X_old, y_old = silos_data[silo_id]
    X_new, y_new = generate_silo_data(seed=9999, n_samples=500, variation=0.15)
    
    model.eval()
    with torch.no_grad():
        loss_old_before = torch.nn.MSELoss()(model(X_old, silo_id), y_old).item()
        loss_new_before = torch.nn.MSELoss()(model(X_new, silo_id), y_new).item()
    
    print(f"\n[2] Antes de actualización:")
    print(f"    Antiguos (var=0.05): {loss_old_before:.6f}")
    print(f"    Nuevos   (var=0.15): {loss_new_before:.6f}")
    
    print("\n[3] Actualizando solo la cabeza especializada (shared congelado)...")
    update_losses = update_for_new_domain(model, silo_id, X_new, y_new, X_old, y_old,
                                         steps=150, lr=0.001, replay_weight=0.5)
    
    with torch.no_grad():
        loss_old_after = torch.nn.MSELoss()(model(X_old, silo_id), y_old).item()
        loss_new_after = torch.nn.MSELoss()(model(X_new, silo_id), y_new).item()
    
    print(f"\n[4] Después de actualización:")
    print(f"    Antiguos: {loss_old_before:.6f} → {loss_old_after:.6f} (variación {(loss_old_after/loss_old_before - 1)*100:.2f}%)")
    print(f"    Nuevos:   {loss_new_before:.6f} → {loss_new_after:.6f} (mejora {(loss_new_before-loss_new_after)/loss_new_before*100:.2f}%)")
    
    if loss_old_after < loss_old_before * 1.05:
        verdict = "✅ OLVIDO DESPRECIABLE (<5%)"
    elif loss_old_after < loss_old_before * 1.10:
        verdict = "⚠️ OLVIDO MODERADO (5-10%)"
    else:
        verdict = "❌ OLVIDO SIGNIFICATIVO (>10%)"
    
    if loss_new_after < loss_new_before:
        verdict += " + MEJORA EN NUEVOS DATOS"
    else:
        verdict += " + NO HAY MEJORA (NUEVOS EMPEORAN)"
    
    print(f"\n🎯 VEREDICTO: {verdict}")

if __name__ == "__main__":
    test_shift()
