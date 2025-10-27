#!/usr/bin/env python3
"""
Taller Clase 5: Detección de Objetos - EfficientNet
Modelo: EfficientNet-B0/B3 como extractor de características CONGELADO
"""
from __future__ import annotations

import glob
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.io import loadmat
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

# Configuración del modelo EfficientNet OPTIMIZADO
# Cambiar a "EfficientNet-B3" si quieres usar EfficientNet-B3
MODEL_NAME = 'EfficientNet-B0'
BATCH_SIZE = 32  # Batch size más grande porque es más rápido
LEARNING_RATE = 1e-3  # Learning rate más alto porque solo entrena la cabeza
EPOCHS = 20  # Menos épocas porque converge más rápido
PATIENCE = 8  # Menos paciencia porque converge más rápido
MIN_DELTA = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Dispositivo: {device}")
print('📊 Configuración OPTIMIZADA:')
print(f"   - Modelo: {MODEL_NAME}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - Learning Rate: {LEARNING_RATE}")
print(f"   - Épocas: {EPOCHS}")
print(f"   - Early Stopping: {PATIENCE} épocas")
print('   - BACKBONE CONGELADO: Solo entrena la cabeza de predicción')

# ============================================================================
# ## 1. Descargar dataset y etiquetas en imágenes (x_top, y_top)
# ============================================================================


class BoundingBoxDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Cargar anotación - CORREGIDO
        ann_path = self.annotation_paths[idx]
        annotation = loadmat(ann_path)
        box_coord = annotation['box_coord'][0]  # Formato: [y1, y2, x1, x2]

        # CORRECCIÓN CRÍTICA: Convertir formato MATLAB [y1, y2, x1, x2] a
        # [x1, y1, x2, y2]
        y1, y2, x1, x2 = box_coord
        correct_coords = [x1, y1, x2, y2]  # [x_left, y_top, x_right, y_bottom]

        # Normalizar coordenadas por tamaño de imagen
        h, w = image.shape[:2]
        normalized_coords = np.array(
            [
                correct_coords[0] / w,  # x_left
                correct_coords[1] / h,  # y_top
                correct_coords[2] / w,  # x_right
                correct_coords[3] / h,   # y_bottom
            ], dtype=np.float32,
        )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(normalized_coords)


def load_dataset():
    """Cargar dataset Caltech-101 con anotaciones"""
    base_path = (
        '/Users/andrestrepo/Documents/repos_personal/vision_aplicada/'
        'taller_clase_5/caltech-101'
    )
    categories = [
        d for d in os.listdir(os.path.join(base_path, '101_ObjectCategories'))
        if os.path.isdir(os.path.join(base_path, '101_ObjectCategories', d))
    ]

    image_paths = []
    annotation_paths = []

    for category in categories:
        img_dir = os.path.join(base_path, '101_ObjectCategories', category)
        ann_dir = os.path.join(base_path, 'Annotations', category)

        if os.path.exists(ann_dir):
            img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
            for img_file in img_files:
                img_name = os.path.basename(img_file)
                ann_name = img_name.replace(
                    'image', 'annotation',
                ).replace('.jpg', '.mat')
                ann_file = os.path.join(ann_dir, ann_name)

                if os.path.exists(ann_file):
                    image_paths.append(img_file)
                    annotation_paths.append(ann_file)

    print(f"📁 Dataset cargado: {len(image_paths)} muestras")
    print(f"📂 Categorías: {len(categories)}")

    return image_paths, annotation_paths

# ============================================================================
# ## 3. Modelo: Extractor de características + Cabeza
# (regresión de 4 variables)
# ============================================================================


class EfficientNetBoundingBoxRegressor(nn.Module):
    def __init__(self, backbone='efficientnet_b0', num_classes=4):
        super().__init__()

        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_size = 1280
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=True)
            feature_size = 1536
        else:
            raise ValueError(f"Backbone {backbone} no soportado")

        # Remover la capa de clasificación original
        self.backbone.classifier = nn.Identity()

        # CONGELAR EL BACKBONE - OPTIMIZACIÓN CRÍTICA
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Regresor optimizado para EfficientNet - MEJORADO
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.SiLU(),  # Swish activation (mejor para EfficientNet)
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),  # Más dropout para regularización
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        coords = self.regressor(features)
        coords = self.sigmoid(coords)
        return coords


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            # Pesos más balanceados para mejor entrenamiento
            self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Todos iguales
        else:
            self.weights = weights

    def forward(self, predictions, targets):
        mse_per_coord = torch.mean((predictions - targets) ** 2, dim=0)
        weighted_mse = torch.sum(
            self.weights.to(
                predictions.device,
            ) * mse_per_coord,
        )
        return weighted_mse

# ============================================================================
# ## 4. Entrenamiento
# ============================================================================


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs, device, patience=8, min_delta=1e-4,
):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print(
        f"🚀 Iniciando entrenamiento OPTIMIZADO de {MODEL_NAME} por {epochs} "
        f"épocas con early stopping (patience={patience})",
    )
    print(
        '⚡ OPTIMIZACIÓN: Solo entrena la cabeza de predicción '
        '(backbone congelado)',
    )
    print('=' * 80)

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % max(1, num_batches // 10) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(
                    f"\rEpoch {epoch+1}/{epochs} - "
                    f"Batch {batch_idx+1}/{num_batches} "
                    f"({progress:.1f}%) - Loss: {loss.item():.4f}", end='',
                )

        # Validación
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        print(
            f"\rEpoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} "
            f"| Patience: {patience_counter}/{patience}",
        )

        if patience_counter >= patience:
            print(
                f"\n⏹️ Early stopping activado en época {epoch+1}. "
                f"No hay mejora en {patience} épocas consecutivas.",
            )
            print(f"🏆 Mejor pérdida de validación: {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)
            break

    print('=' * 80)
    print(f"✅ Entrenamiento completado. Total de épocas: {len(train_losses)}")
    print(f"🏆 Mejor pérdida de validación: {best_val_loss:.4f}")

    return train_losses, val_losses

# ============================================================================
# ### 4.1 Visualizar resultados entrenamiento
# ============================================================================


def plot_training_results(train_losses, val_losses):
    """Visualizar resultados del entrenamiento"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Pérdida durante el entrenamiento - {MODEL_NAME} OPTIMIZADO')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Pérdida (escala log) - {MODEL_NAME} OPTIMIZADO')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ============================================================================
# ## 5. Evaluación
# ============================================================================


def evaluate_model_detailed(model, val_loader, device):
    """Evaluación detallada del modelo"""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = nn.MSELoss()(outputs, targets)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / num_samples
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(avg_loss)
    mae_per_coord = np.mean(np.abs(all_predictions - all_targets), axis=0)

    return {
        'mse': avg_loss,
        'rmse': rmse,
        'mae': mae,
        'mae_per_coordinate': mae_per_coord,
        'predictions': all_predictions,
        'targets': all_targets,
    }


def print_evaluation_results(evaluation_results):
    """Imprimir resultados de evaluación"""
    print(
        f"\n📊 Resultados de evaluación - {MODEL_NAME} OPTIMIZADO:",
    )
    print(f"   - MSE: {evaluation_results['mse']:.6f}")
    print(f"   - RMSE: {evaluation_results['rmse']:.6f}")
    print(f"   - MAE: {evaluation_results['mae']:.6f}")
    print('\n📏 Error absoluto medio por coordenada:')
    coord_names = ['x_left', 'y_top', 'x_right', 'y_bottom']  # CORREGIDO
    for i, coord_name in enumerate(coord_names):
        print(
            f"   - {coord_name}: "
            f"{evaluation_results['mae_per_coordinate'][i]:.6f}",
        )

    avg_error_percentage = evaluation_results['mae'] * 100
    print(
        f"\n📈 Error promedio: {avg_error_percentage:.2f}% de las "
        f"coordenadas normalizadas",
    )

# ============================================================================
# ## 6. Predicciones evaluación
# ============================================================================


def visualize_predictions(
    model, dataset, num_samples=5, device='cpu',
):
    """Visualizar predicciones del modelo con mejor distribución y
    legibilidad"""
    model.eval()

    # Configuración mejorada para mejor distribución
    if num_samples <= 2:
        fig, axes = plt.subplots(1, num_samples, figsize=(16, 6))
    elif num_samples <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

    if num_samples == 1:
        axes = [axes]

    # Configurar estilo de texto
    plt.rcParams.update({'font.size': 10, 'font.weight': 'bold'})

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, target = dataset[idx]

            image_batch = image.unsqueeze(0).to(device)
            prediction = model(image_batch).cpu().numpy()[0]

            # CORRECCIÓN CRÍTICA: Desnormalizar imagen correctamente
            img_np = image.permute(1, 2, 0).numpy()
            # Desnormalizar usando los valores correctos de ImageNet
            img_np = (
                img_np * np.array([0.229, 0.224, 0.225]) +
                np.array([0.485, 0.456, 0.406])
            )
            img_np = np.clip(img_np, 0, 1)

            h, w = img_np.shape[:2]
            pred_clipped = np.clip(prediction, 0, 1)
            target_clipped = np.clip(target.numpy(), 0, 1)

            # CORRECCIÓN CRÍTICA: Usar formato correcto
            # [x_left, y_top, x_right, y_bottom]
            pred_coords = pred_clipped * [w, h, w, h]
            target_coords = target_clipped * [w, h, w, h]

            img_with_boxes = img_np.copy()

            # Dibujar predicción (verde) - CORREGIDO
            x1_pred, y1_pred, x2_pred, y2_pred = pred_coords.astype(int)
            x1_pred, y1_pred = max(0, x1_pred), max(0, y1_pred)
            x2_pred, y2_pred = min(w-1, x2_pred), min(h-1, y2_pred)
            cv2.rectangle(
                img_with_boxes, (x1_pred, y1_pred),
                (x2_pred, y2_pred), (0, 255, 0), 3,
            )

            # Dibujar ground truth (rojo) - CORREGIDO
            x1_true, y1_true, x2_true, y2_true = target_coords.astype(int)
            x1_true, y1_true = max(0, x1_true), max(0, y1_true)
            x2_true, y2_true = min(w-1, x2_true), min(h-1, y2_true)
            cv2.rectangle(
                img_with_boxes, (x1_true, y1_true),
                (x2_true, y2_true), (255, 0, 0), 3,
            )

            error_normalized = np.mean(np.abs(prediction - target.numpy()))
            error_pixels = np.mean(np.abs(pred_coords - target_coords))

            def calculate_iou(box1, box2):
                x1, y1, x2, y2 = box1
                x1_true, y1_true, x2_true, y2_true = box2

                x_left = max(x1, x1_true)
                y_top = max(y1, y1_true)
                x_right = min(x2, x2_true)
                y_bottom = min(y2, y2_true)

                if x_right < x_left or y_bottom < y_top:
                    return 0.0

                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (x2_true - x1_true) * (y2_true - y1_true)
                union = area1 + area2 - intersection

                return intersection / union if union > 0 else 0.0

            iou = calculate_iou(pred_coords, target_coords)

            # Mostrar imagen
            axes[i].imshow(img_with_boxes)

            # Título mejorado con mejor formato
            title_text = f'{MODEL_NAME} - Predicción {i+1}'
            subtitle_text = 'Verde: Predicción | Rojo: Real'
            metrics_text = (
                f'Error norm: {error_normalized:.3f} | '
                f'Error px: {error_pixels:.1f} | IoU: {iou:.3f}'
            )

            # Título principal
            axes[i].set_title(
                title_text, fontsize=12,
                fontweight='bold', pad=10,
            )

            # Subtítulo con métricas
            axes[i].text(
                0.5, -0.15, subtitle_text, transform=axes[i].transAxes,
                ha='center', fontsize=10, color='blue', fontweight='bold',
            )
            axes[i].text(
                0.5, -0.25, metrics_text, transform=axes[i].transAxes,
                ha='center', fontsize=9, color='darkgreen',
            )

            axes[i].axis('off')

    # Ocultar ejes vacíos si los hay
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)

    # Título general de la figura
    fig.suptitle(
        f'🎯 Predicciones de Bounding Box - {MODEL_NAME}',
        fontsize=16, fontweight='bold', y=0.95,
    )

    # Ajustar layout con más espacio
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()
