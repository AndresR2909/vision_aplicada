from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class JigsawPuzzleDataset(Dataset):
    """
    Dataset para tarea de pretexto Jigsaw Puzzle
    Divide la imagen en 9 parches (3x3) y los reordena según
    permutaciones predefinidas
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # Definir permutaciones predefinidas para el rompecabezas 3x3
        # Cada permutación es un reordenamiento de [0,1,2,3,4,5,6,7,8]
        self.permutations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Original
            [1, 0, 2, 4, 3, 5, 7, 6, 8],  # Permutación 1
            [2, 1, 0, 5, 4, 3, 8, 7, 6],  # Permutación 2
            [3, 4, 5, 0, 1, 2, 6, 7, 8],  # Permutación 3
            [4, 3, 5, 1, 0, 2, 7, 6, 8],  # Permutación 4
            [5, 4, 3, 2, 1, 0, 8, 7, 6],  # Permutación 5
            [6, 7, 8, 0, 1, 2, 3, 4, 5],  # Permutación 6
            [7, 6, 8, 1, 0, 2, 4, 3, 5],  # Permutación 7
            [8, 7, 6, 2, 1, 0, 5, 4, 3],  # Permutación 8
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Obtener imagen original
        image, _ = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        # Convertir a numpy para manipulación
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # Seleccionar permutación aleatoria
        perm_idx = random.randint(0, len(self.permutations) - 1)
        permutation = self.permutations[perm_idx]

        # Crear imagen con parches reordenados
        jigsaw_image = self.create_jigsaw_puzzle(image, permutation)

        return jigsaw_image, perm_idx

    def create_jigsaw_puzzle(self, image, permutation):
        """
        Crea el rompecabezas reordenando los parches según la permutación
        """
        # Asumimos imagen de forma (C, H, W)
        C, H, W = image.shape

        # Calcular tamaño de cada parche
        patch_h = H // 3
        patch_w = W // 3

        # Extraer parches
        patches = []
        for i in range(3):
            for j in range(3):
                start_h = i * patch_h
                end_h = (i + 1) * patch_h
                start_w = j * patch_w
                end_w = (j + 1) * patch_w
                patch = image[:, start_h:end_h, start_w:end_w]
                patches.append(patch)

        # Reordenar parches según la permutación
        reordered_patches = [patches[permutation[i]] for i in range(9)]

        # Reconstruir imagen
        jigsaw_image = np.zeros_like(image)
        for i in range(3):
            for j in range(3):
                patch_idx = i * 3 + j
                start_h = i * patch_h
                end_h = (i + 1) * patch_h
                start_w = j * patch_w
                end_w = (j + 1) * patch_w
                jigsaw_image[
                    :, start_h:end_h,
                    start_w:end_w,
                ] = reordered_patches[patch_idx]

        return torch.tensor(jigsaw_image, dtype=torch.float32)

    @staticmethod
    def visualize_jigsaw_puzzle(dataset, num_samples=4):
        """Visualizar ejemplos de rompecabezas generados"""

        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))

        for i in range(num_samples):
            # Obtener muestra
            jigsaw_img, perm_idx = dataset[i]

            # Convertir tensor a imagen para visualización
            if isinstance(jigsaw_img, torch.Tensor):
                img = jigsaw_img.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (
                    img.max() -
                    img.min()
                )  # Normalizar a [0,1]
            else:
                img = jigsaw_img

            # Mostrar imagen original (primera fila)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Jigsaw Puzzle\nPermutación: {perm_idx}')
            axes[0, i].axis('off')

            # Mostrar parches individuales (segunda fila)
            patch_h, patch_w = img.shape[0] // 3, img.shape[1] // 3
            for j in range(9):
                row = j // 3
                col = j % 3
                start_h = row * patch_h
                end_h = (row + 1) * patch_h
                start_w = col * patch_w
                end_w = (col + 1) * patch_w
                patch = img[start_h:end_h, start_w:end_w]

                # Crear subplot para cada parche
                if j == 0:
                    axes[1, i].imshow(patch)
                    axes[1, i].set_title('Parches individuales')
                    axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
