from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model, train_loader, val_loader, num_epochs, learning_rate, device,
    patience=5, min_delta=0.001,
):
    """Función para entrenar el modelo con early stopping

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        num_epochs: Número máximo de épocas
        learning_rate: Tasa de aprendizaje
        device: Dispositivo (cuda/cpu)
        patience: Número de épocas sin mejora antes de parar (default: 5)
        min_delta: Cambio mínimo para considerar mejora (default: 0.001)
    """

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Variables para early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print('Iniciando entrenamiento...')
    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch {epoch+1}/{num_epochs}, '
                    f'Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}',
                )

        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Calcular métricas
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping logic
        if val_loss_avg < best_val_loss - min_delta:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # Guardar el mejor estado del modelo
            best_model_state = model.state_dict().copy()
            # Guardar el modelo cuando es el mejor
            torch.save(model.state_dict(), 'vit_jigsaw_model.pth')
            print(f'  ✅ Nueva mejor pérdida de validación: {val_loss_avg:.4f}')
            print('  💾 Modelo guardado como mejor modelo')
        else:
            patience_counter += 1
            print(f'  ⏳ Sin mejora por {patience_counter}/{patience} épocas')

        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(
            f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%',
        )
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)

        # Verificar early stopping
        if patience_counter >= patience:
            print(f'\n🛑 Early stopping activado después de {epoch+1} épocas')
            print(f'   Mejor pérdida de validación: {best_val_loss:.4f}')
            print('   Restaurando mejor modelo...')
            # Restaurar el mejor modelo
            model.load_state_dict(best_model_state)
            break

    # Asegurar que el mejor modelo esté guardado al final
    if best_model_state is not None:
        # Restaurar el mejor modelo y guardarlo una vez más para asegurar
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'vit_jigsaw_model.pth')
        print(
            '\n✅ Entrenamiento completado. Mejor modelo guardado como '
            "'vit_jigsaw_model.pth'",
        )
        print(f"   Mejor pérdida de validación: {best_val_loss:.4f}")
    else:
        # Si no hubo mejora, guardar el modelo actual
        torch.save(model.state_dict(), 'vit_jigsaw_model.pth')
        print(
            '\n✅ Entrenamiento completado. Modelo final guardado como '
            "'vit_jigsaw_model.pth'",
        )

    return train_losses, val_losses, train_accs, val_accs


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Visualizar el historial de entrenamiento"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de pérdida
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Gráfico de precisión
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
