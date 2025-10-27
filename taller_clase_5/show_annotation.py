from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat


def show_annotation(imgfile, annotation_file):
    """
    Visualiza una imagen con sus anotaciones (bounding box y contorno)

    Args:
        imgfile (str): Ruta al archivo de imagen
        annotation_file (str): Ruta al archivo de anotación (.mat)

    Basado en el código original de Fei-Fei Li - Noviembre 2004
    """

    # Parámetros de visualización
    MEDFONT = 18

    # Cargar los datos anotados
    try:
        annotation_data = loadmat(annotation_file)
        print(
            f"Claves disponibles en anotación: {list(annotation_data.keys())}",
        )

        # Verificar si existe box_coord
        if 'box_coord' in annotation_data:
            box_coord = annotation_data['box_coord']
            print(
                f"box_coord shape: {box_coord.shape}, contenido: {box_coord}",
            )
        else:
            print("No se encontró 'box_coord' en la anotación")
            return

        # Verificar si existe obj_contour
        if 'obj_contour' in annotation_data:
            obj_contour = annotation_data['obj_contour']
            print(f"obj_contour shape: {obj_contour.shape}")
        else:
            print("No se encontró 'obj_contour' en la anotación")
            obj_contour = np.array([])

    except Exception as e:
        print(f"Error cargando anotaciones: {e}")
        return

    # Leer y mostrar la imagen
    try:
        ima = Image.open(imgfile)
        if ima.mode == 'L':  # Imagen en escala de grises
            ima = ima.convert('RGB')

        # Convertir a numpy array
        ima_array = np.array(ima)

        # Crear la figura
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(ima_array)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Equivalente a axis ij en MATLAB

        # Mostrar bounding box
        # Verificar el formato de box_coord
        if box_coord.size >= 4:
            # Formato esperado: [y1, y2, x1, x2] en MATLAB
            if box_coord.ndim == 1:
                # Esquina superior izquierda
                x1, y1 = box_coord[2], box_coord[0]
                width = box_coord[3] - box_coord[2]  # Ancho
                height = box_coord[1] - box_coord[0]  # Alto
            else:
                # Si es 2D, tomar la primera fila
                # box_coord[0] = [y1, y2, x1, x2]
                x1, y1 = box_coord[0, 2], box_coord[0, 0]
                width = box_coord[0, 3] - box_coord[0, 2]
                height = box_coord[0, 1] - box_coord[0, 0]
        else:
            print(
                f"Formato de box_coord no soportado. "
                f"Tamaño: {box_coord.size}, forma: {box_coord.shape}",
            )
            return

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=5, edgecolor='yellow', facecolor='none',
        )
        ax.add_patch(rect)

        # Mostrar contorno del objeto
        if obj_contour.size > 0 and obj_contour.shape[0] >= 2:
            # Ajustar coordenadas del contorno con el offset del bounding box
            if box_coord.ndim == 1:
                contour_x = obj_contour[0, :] + box_coord[2]
                contour_y = obj_contour[1, :] + box_coord[0]
            else:
                contour_x = obj_contour[0, :] + box_coord[0, 2]
                contour_y = obj_contour[1, :] + box_coord[0, 0]

            # Dibujar el contorno
            for cc in range(obj_contour.shape[1]):
                if cc < obj_contour.shape[1] - 1:
                    ax.plot(
                        [contour_x[cc], contour_x[cc+1]],
                        [contour_y[cc], contour_y[cc+1]],
                        'r', linewidth=4,
                    )
                else:
                    # Conectar el último punto con el primero
                    ax.plot(
                        [contour_x[cc], contour_x[0]],
                        [contour_y[cc], contour_y[0]],
                        'r', linewidth=4,
                    )

        # Configurar la ventana
        ax.set_title(os.path.basename(imgfile), fontsize=MEDFONT)
        ax.axis('off')  # Ocultar ejes

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error procesando imagen: {e}")


def show_annotation_batch(img_dir, annotation_dir, category, num_samples=5):
    """
    Muestra múltiples anotaciones de una categoría específica

    Args:
        img_dir (str): Directorio de imágenes
        annotation_dir (str): Directorio de anotaciones
        category (str): Categoría a visualizar
        num_samples (int): Número de muestras a mostrar
    """

    category_img_dir = os.path.join(img_dir, category)
    category_ann_dir = os.path.join(annotation_dir, category)

    if not os.path.exists(category_img_dir) or not os.path.exists(
        category_ann_dir,
    ):
        print(f"Directorio no encontrado para la categoría: {category}")
        return

    # Obtener lista de archivos de imagen
    img_files = [f for f in os.listdir(category_img_dir) if f.endswith('.jpg')]
    img_files.sort()

    # Mostrar las primeras num_samples imágenes
    for i, img_file in enumerate(img_files[:num_samples]):
        img_path = os.path.join(category_img_dir, img_file)

        # Buscar archivo de anotación correspondiente
        base_name = os.path.splitext(img_file)[0]
        ann_file = f"annotation_{base_name.split('_')[-1]}.mat"
        ann_path = os.path.join(category_ann_dir, ann_file)

        if os.path.exists(ann_path):
            print(f"\nMostrando: {img_file}")
            show_annotation(img_path, ann_path)
        else:
            print(f"Anotación no encontrada para: {img_file}")

# Función de utilidad para explorar el dataset


def explore_dataset(base_path):
    """
    Explora la estructura del dataset Caltech-101

    Args:
        base_path (str): Ruta base del dataset descomprimido
    """

    img_dir = os.path.join(base_path, '101_ObjectCategories')
    ann_dir = os.path.join(base_path, 'Annotations')

    print('=== Explorando Dataset Caltech-101 ===')
    print(f"Directorio de imágenes: {img_dir}")
    print(f"Directorio de anotaciones: {ann_dir}")

    if os.path.exists(img_dir):
        categories = sorted(os.listdir(img_dir))
        print(f"\nCategorías disponibles ({len(categories)}):")
        for i, cat in enumerate(categories[:10]):  # Mostrar primeras 10
            cat_path = os.path.join(img_dir, cat)
            if os.path.isdir(cat_path):
                img_count = len([
                    f for f in os.listdir(
                        cat_path,
                    ) if f.endswith('.jpg')
                ])
                print(f"  {i+1:2d}. {cat:<20} ({img_count:3d} imágenes)")

        if len(categories) > 10:
            print(f"  ... y {len(categories) - 10} categorías más")

    if os.path.exists(ann_dir):
        ann_categories = sorted(os.listdir(ann_dir))
        print(f"\nCategorías con anotaciones ({len(ann_categories)}):")
        for i, cat in enumerate(ann_categories[:10]):
            if os.path.isdir(os.path.join(ann_dir, cat)):
                ann_count = len([
                    f for f in os.listdir(
                        os.path.join(ann_dir, cat),
                    ) if f.endswith('.mat')
                ])
                print(f"  {i+1:2d}. {cat:<20} ({ann_count:3d} anotaciones)")
