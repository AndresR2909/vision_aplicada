from __future__ import annotations

import multiprocessing
import os
import pickle
import warnings

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel
from transformers import CLIPProcessor

# Configurar para evitar warnings de multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Configurar multiprocessing para macOS
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Ya está configurado

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='multiprocessing.resource_tracker',
)
warnings.filterwarnings('ignore', category=FutureWarning)


class CLIPImageRetrievalSystem:
    """
    Sistema de recuperación de imágenes usando CLIP y FAISS
    """

    def __init__(self, model_id='openai/clip-vit-base-patch32', device=None):
        """
        Inicializar el sistema de recuperación

        Args:
            model_id: ID del modelo CLIP a usar
            device: Dispositivo para el modelo (auto-detecta si es None)
        """
        self.model_id = model_id
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Cargar modelo y procesador CLIP
        print(f"Cargando modelo CLIP: {model_id}")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        # Inicializar índices FAISS
        self.image_index = None
        self.text_index = None
        self.image_embeddings = None
        self.text_embeddings = None
        self.metadata = []

        print(f"Modelo cargado en dispositivo: {self.device}")

    def extract_image_embeddings(self, images, batch_size=32):
        """
        Extraer embeddings de imágenes usando CLIP

        Args:
            images: Lista de imágenes PIL
            batch_size: Tamaño del lote para procesamiento

        Returns:
            numpy array con embeddings normalizados
        """
        print(f"Extrayendo embeddings de {len(images)} imágenes...")

        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size)):
            batch_images = images[i:i+batch_size]

            # Procesar lote de imágenes
            inputs = self.processor(
                text=None,
                images=batch_images,
                return_tensors='pt',
                padding=True,
            )['pixel_values'].to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_image_features(inputs)
                # Normalizar embeddings
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenar todos los embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"Embeddings de imágenes extraídos: {embeddings.shape}")

        return embeddings

    def extract_text_embeddings(self, texts, batch_size=32):
        """
        Extraer embeddings de texto usando CLIP

        Args:
            texts: Lista de textos
            batch_size: Tamaño del lote para procesamiento

        Returns:
            numpy array con embeddings normalizados
        """
        print(f"Extrayendo embeddings de {len(texts)} textos...")

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]

            # Procesar lote de textos
            inputs = self.processor(
                text=batch_texts,
                images=None,
                return_tensors='pt',
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
                # Normalizar embeddings
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenar todos los embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"Embeddings de texto extraídos: {embeddings.shape}")

        return embeddings

    def build_index(self, dataset, embedding_dim=512):
        """
        Construir índices FAISS para el dataset

        Args:
            dataset: Dataset de Hugging Face con imágenes y metadatos
            embedding_dim: Dimensión de los embeddings
        """
        print('Construyendo índices FAISS...')

        # Extraer embeddings de imágenes
        images = [example['image'] for example in dataset]
        self.image_embeddings = self.extract_image_embeddings(images)

        # Extraer embeddings de texto (nombres de categorías)
        texts = [example['category_name'] for example in dataset]
        self.text_embeddings = self.extract_text_embeddings(texts)

        # Crear índices FAISS
        # Inner Product para similitud coseno
        self.image_index = faiss.IndexFlatIP(embedding_dim)
        self.text_index = faiss.IndexFlatIP(embedding_dim)

        # Agregar embeddings a los índices
        self.image_index.add(self.image_embeddings.astype('float32'))
        self.text_index.add(self.text_embeddings.astype('float32'))

        # Guardar metadatos
        self.metadata = [
            {
                'index': i,
                'category_id': example['category_id'],
                'category_name': example['category_name'],
                'image': example['image'],
            }
            for i, example in enumerate(dataset)
        ]

        print('Índices construidos:')
        print(f"  - Imágenes: {self.image_index.ntotal} embeddings")
        print(f"  - Textos: {self.text_index.ntotal} embeddings")

    def search_by_text(self, query_text, k=5):
        """
        Buscar imágenes similares usando texto de consulta

        Args:
            query_text: Texto de consulta
            k: Número de resultados a retornar

        Returns:
            Lista de resultados con metadatos
        """
        if self.text_index is None:
            raise ValueError(
                'Índice no construido. Llama a build_index() primero.',
            )

        # Extraer embedding del texto de consulta
        query_embedding = self.extract_text_embeddings([query_text])

        # Buscar en el índice de imágenes
        scores, indices = self.image_index.search(
            query_embedding.astype('float32'), k,
        )

        # Preparar resultados
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                results.append(result)

        return results

    def search_by_image(self, query_image, k=5):
        """
        Buscar imágenes similares usando una imagen de consulta

        Args:
            query_image: Imagen PIL de consulta
            k: Número de resultados a retornar

        Returns:
            Lista de resultados con metadatos
        """
        if self.image_index is None:
            raise ValueError(
                'Índice no construido. Llama a build_index() primero.',
            )

        # Extraer embedding de la imagen de consulta
        query_embedding = self.extract_image_embeddings([query_image])

        # Buscar en el índice de imágenes
        scores, indices = self.image_index.search(
            query_embedding.astype('float32'), k,
        )

        # Preparar resultados
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                results.append(result)

        return results

    def save_index(self, filepath):
        """
        Guardar índices y metadatos en disco

        Args:
            filepath: Ruta base para guardar archivos
        """
        print(f"Guardando índices en {filepath}...")

        # Guardar índices FAISS
        faiss.write_index(self.image_index, f"{filepath}_image.index")
        faiss.write_index(self.text_index, f"{filepath}_text.index")

        # Guardar embeddings
        np.save(f"{filepath}_image_embeddings.npy", self.image_embeddings)
        np.save(f"{filepath}_text_embeddings.npy", self.text_embeddings)

        # Guardar metadatos
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)

        print('Índices guardados exitosamente')

    def load_index(self, filepath):
        """
        Cargar índices y metadatos desde disco

        Args:
            filepath: Ruta base de los archivos guardados
        """
        print(f"Cargando índices desde {filepath}...")

        # Cargar índices FAISS
        self.image_index = faiss.read_index(f"{filepath}_image.index")
        self.text_index = faiss.read_index(f"{filepath}_text.index")

        # Cargar embeddings
        self.image_embeddings = np.load(
            f"{filepath}_image_embeddings.npy", allow_pickle=True,
        )
        self.text_embeddings = np.load(
            f"{filepath}_text_embeddings.npy", allow_pickle=True,
        )

        # Cargar metadatos
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

        print('Índices cargados exitosamente')

    # Funciones auxiliares para visualización de resultados

    def search_and_visualize_text(self, query_text, k=8):
        """
        Buscar por texto y visualizar resultados

        Args:
            query_text: Texto de consulta
            k: Número de resultados
        """
        print(f"Buscando imágenes para: '{query_text}'")
        results = self.search_by_text(query_text, k=k)

        print(f"\nEncontrados {len(results)} resultados:")
        for i, result in enumerate(results):
            print(
                f"  {i+1}. {result['category_name']} "
                f"(Score: {result['score']:.3f})",
            )

        CLIPImageRetrievalSystem.visualize_search_results(
            results, query_text=query_text, max_results=k,
        )

    def search_and_visualize_image(self, query_image, k=8):
        """
        Buscar por imagen y visualizar resultados

        Args:
            query_image: Imagen PIL de consulta
            k: Número de resultados
        """
        print('Buscando imágenes similares...')
        results = self.search_by_image(query_image, k=k)

        print(f"\nEncontrados {len(results)} resultados:")
        for i, result in enumerate(results):
            print(
                f"  {i+1}. {result['category_name']} "
                f"(Score: {result['score']:.3f})",
            )

        CLIPImageRetrievalSystem.visualize_search_results(
            results, query_image=query_image, max_results=k,
        )

    @staticmethod
    def visualize_search_results(
        results, query_text=None, query_image=None, max_results=8,
    ):
        """
        Visualizar resultados de búsqueda

        Args:
            results: Lista de resultados de búsqueda
            query_text: Texto de consulta (opcional)
            query_image: Imagen de consulta (opcional)
            max_results: Máximo número de resultados a mostrar
        """
        num_results = min(len(results), max_results)

        # Calcular layout de subplots
        if query_image is not None:
            cols = 4
            rows = (num_results + 3) // 4  # +3 para redondear hacia arriba
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
        else:
            cols = 4
            rows = (num_results + 3) // 4
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)

        # Mostrar imagen de consulta si existe
        if query_image is not None:
            axes[0, 0].imshow(query_image)
            axes[0, 0].set_title(
                'Imagen de Consulta',
                fontsize=12, fontweight='bold',
            )
            axes[0, 0].axis('off')
            start_idx = 1
        else:
            start_idx = 0

        # Mostrar resultados
        for i, result in enumerate(results[:num_results]):
            row = (i + start_idx) // cols
            col = (i + start_idx) % cols

            if row < rows and col < cols:
                axes[row, col].imshow(result['image'])
                title = (
                    f"#{result['rank']}: {result['category_name']}\n"
                    f"Score: {result['score']:.3f}"
                )
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis('off')

        # Ocultar ejes vacíos
        for i in range(num_results + start_idx, rows * cols):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')

        # Título general
        if query_text:
            fig.suptitle(
                f"Resultados para: '{query_text}'",
                fontsize=16, fontweight='bold',
            )
        elif query_image is not None:
            fig.suptitle(
                'Resultados por similitud de imagen',
                fontsize=16, fontweight='bold',
            )

        plt.tight_layout()
        plt.show()
