# Evaluación completa del modelo con métricas adicionales
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def evaluate_model_comprehensive(model, test_loader, device, class_names=None):
    """
    Evaluación completa del modelo con métricas adicionales
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    print('🔍 Evaluando modelo en conjunto de test...')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Obtener predicciones
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"Procesando batch {batch_idx}/{len(test_loader)}")

    # Convertir a arrays numpy
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
    precision_macro = precision_score(
        all_targets, all_predictions, average='macro',
    )
    recall_macro = recall_score(all_targets, all_predictions, average='macro')

    print('\n📊 MÉTRICAS DE EVALUACIÓN:')
    print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - F1-Score (Macro): {f1_macro:.4f}")
    print(f"  - F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"  - Precision (Macro): {precision_macro:.4f}")
    print(f"  - Recall (Macro): {recall_macro:.4f}")

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
    """
    Visualizar matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.show()

    return cm


def plot_classification_report(y_true, y_pred, class_names=None):
    """
    Visualizar reporte de clasificación
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True,
    )

    # Crear DataFrame para visualización
    df_report = pd.DataFrame(report).transpose()

    # Visualizar métricas por clase
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de barras para precision, recall, f1-score
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax1.bar(
            x + i*width, [
                df_report.loc[class_name, metric]
                for class_name in class_names
            ],
            width, label=metric,
        )

    ax1.set_xlabel('Clases')
    ax1.set_ylabel('Score')
    ax1.set_title('Métricas por Clase')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Heatmap de métricas
    metrics_data = df_report.loc[class_names, metrics].values
    im = ax2.imshow(metrics_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metrics)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_title('Heatmap de Métricas')

    # Añadir valores en el heatmap
    for i in range(len(class_names)):
        for j in range(len(metrics)):
            ax2.text(
                j, i, f'{metrics_data[i, j]:.3f}',
                ha='center', va='center', color='black',
            )

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()

    return report


def visualize_predictions(
    model, test_loader, device, num_samples=8, class_names=None,
):
    """
    Visualizar predicciones del modelo con ejemplos
    """
    model.eval()

    # Obtener algunas muestras
    data_iter = iter(test_loader)
    images, targets = next(data_iter)
    images, targets = images.to(device), targets.to(device)

    # Hacer predicciones
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    # Seleccionar muestras aleatorias
    indices = torch.randperm(len(images))[:num_samples]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = images[idx].cpu()
        target = targets[idx].cpu().item()
        pred = predictions[idx].cpu().item()
        prob = probabilities[idx].cpu().numpy()

        # Desnormalizar imagen para visualización
        img_denorm = img.clone()
        for t, m, s in zip(
            img_denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        ):
            t.mul_(s).add_(m)
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # Mostrar imagen
        axes[i].imshow(img_denorm.permute(1, 2, 0))

        # Título con información
        true_label = class_names[target] if class_names else f'Clase {target}'
        pred_label = class_names[pred] if class_names else f'Clase {pred}'
        confidence = prob[pred] * 100

        color = 'green' if target == pred else 'red'
        axes[i].set_title(
            f'Verdadero: {true_label}\nPredicho: {pred_label}\n'
            f'Confianza: {confidence:.1f}%',
            color=color, fontsize=10,
        )
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_probability_distribution(probabilities, targets, class_names=None):
    """
    Visualizar distribución de probabilidades
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Distribución de confianza para predicciones correctas vs incorrectas
    correct_mask = np.argmax(probabilities, axis=1) == targets
    correct_confidences = np.max(probabilities[correct_mask], axis=1)
    incorrect_confidences = np.max(probabilities[~correct_mask], axis=1)

    ax1.hist(
        correct_confidences, bins=20, alpha=0.7,
        label='Correctas', color='green',
    )
    ax1.hist(
        incorrect_confidences, bins=20, alpha=0.7,
        label='Incorrectas', color='red',
    )
    ax1.set_xlabel('Confianza de Predicción')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Confianza')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Heatmap de probabilidades promedio por clase
    avg_probs = np.zeros((len(class_names), len(class_names)))
    for i, class_name in enumerate(class_names):
        class_mask = targets == i
        if np.sum(class_mask) > 0:
            avg_probs[i] = np.mean(probabilities[class_mask], axis=0)

    im = ax2.imshow(avg_probs, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_title('Probabilidades Promedio por Clase')
    ax2.set_xlabel('Clase Predicha')
    ax2.set_ylabel('Clase Verdadera')

    # Añadir valores en el heatmap
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax2.text(
                j, i, f'{avg_probs[i, j]:.3f}',
                ha='center', va='center', color='black',
            )

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()
