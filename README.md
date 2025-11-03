# Visión Aplicada - Talleres de Machine Learning

Este repositorio contiene una serie de talleres prácticos sobre visión por computadora y machine learning aplicado, desarrollados como parte de un curso de visión aplicada.

## 📋 Estructura del Proyecto

```
vision_aplicada/
├── taller_clase_1/                    # Clasificación de enfermedades en hojas de mango
│   ├── conceptos_taller_1.txt
│   ├── mango_leaf_disease/           # Dataset con 8 categorías de enfermedades
│   └── solucion_taller_1.ipynb
├── taller_clase_2/                    # Clasificación de imágenes con modelos pre-entrenados
│   ├── classes.txt
│   ├── conceptos_taller_2.txt
│   ├── models/                       # Modelos entrenados
│   ├── test/                         # Dataset de prueba
│   ├── train/                        # Dataset de entrenamiento
│   └── taller_clasificacion_imagen_clase_2.ipynb
├── taller_clase_3/                    # Sistema de recuperación texto-imagen
│   ├── __pycache__/
│   ├── conceptos_taller_3.txt
│   ├── evaluate.py
│   ├── image_retrieval_system.py
│   ├── index/                        # Índices de búsqueda
│   ├── requirements.txt
│   ├── run_app.py
│   ├── sistema_recuperacion.png
│   ├── slides-msc-knowledge_compressed.pdf
│   ├── taller_recuperacion_imagenes.ipynb
│   └── ui_streamlit.py
├── taller_clase_4/                    # Entrenamiento auto-supervisado con tareas pretexto
│   ├── __pycache__/
│   ├── conceptos_taller_4.txt
│   ├── eval.py
│   ├── imagenet-mini/                # Dataset ImageNet-mini
│   ├── pretext_tasks.py
│   ├── slides-msc-foundation_compressed.pdf
│   ├── taller_clase_4.ipynb
│   ├── train.py
│   ├── vit_jigsaw_model.pth          # Modelo entrenado
│   └── vit.py
├── taller_clase_5/                    # Detección de objetos con bounding boxes
│   ├── __pycache__/
│   ├── bounding_box_efficientnet.py
│   ├── bounding_box_vanilla_cnn.py
│   ├── caltech-101/                  # Dataset Caltech-101
│   ├── models/                       # Modelos entrenados
│   ├── requirements_annotations.txt
│   ├── show_annotation.py
│   ├── slides-msc-detection-I_compressed.pdf
│   ├── slides-msc-detection-II_compressed.pdf
│   ├── taller_clase_5_efficientnet.ipynb
│   └── taller_clase_5_vanilla_cnn.ipynb
├── taller_clase_6/                    # Segmentación de imágenes con SAM
│   ├── 256_ObjectCategories/         # Dataset Caltech 256
│   ├── annotations_251.airplanes-101_predictor_bbox/# Anotaciones de segmentación
│   ├── sam_vit_h_4b8939.pth          # Modelo SAM pre-entrenado
│   └── taller_clase_6_segmentacion_predictor.ipynb
├── requirements.txt                   # Dependencias del proyecto
└── README.md                         # Este archivo
```

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd vision_aplicada
```

2. Crea un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📚 Talleres Disponibles

### 🔬 Taller 1: Clasificación de Enfermedades en Hojas de Mango
**Archivo:** [`taller_clase_1/solucion_taller_1.ipynb`](taller_clase_1/solucion_taller_1.ipynb)

**Objetivo:** Implementar un modelo de clasificación para predecir enfermedades en hojas de mango usando características HOG (Histogram of Oriented Gradients).

**Características principales:**
- Extracción de características HOG
- Implementación de múltiples algoritmos de clasificación (SVM, Random Forest, MLP)
- Evaluación con métricas de precisión y F1-Score
- Dataset: Mango Leaf Disease con 8 categorías de enfermedades

**Tecnologías:** scikit-image, scikit-learn, OpenCV

---

### 🖼️ Taller 2: Clasificación de Imágenes con Modelos Pre-entrenados
**Archivo:** [`taller_clase_2/taller_clasificacion_imagen_clase_2.ipynb`](taller_clase_2/taller_clasificacion_imagen_clase_2.ipynb)

**Objetivo:** Realizar clasificación de imágenes usando modelos pre-entrenados (ResNet50, MobileNet) como extractores de características.

**Características principales:**
- Exploración del dataset LabelMe-12-50k
- Extracción de características usando modelos pre-entrenados
- Entrenamiento de cabeza de clasificación con AutoGluon
- Evaluación comparativa de diferentes arquitecturas

**Tecnologías:** PyTorch, AutoGluon, torchvision

---

### 🔍 Taller 3: Sistema de Recuperación Texto-Imagen
**Archivo:** [`taller_clase_3/taller_recuperacion_imagenes.ipynb`](taller_clase_3/taller_recuperacion_imagenes.ipynb)

**Objetivo:** Implementar un sistema de recuperación texto-imagen usando el modelo CLIP (Contrastive Language-Image Pre-training).

**Características principales:**
- Implementación de sistema de búsqueda multimodal
- Dataset Caltech 256 con 256 categorías de objetos
- Interfaz interactiva para búsquedas texto-imagen
- Evaluación del rendimiento del sistema de recuperación

**Tecnologías:** CLIP, Transformers, Streamlit, FAISS

---

### 🧩 Taller 4: Entrenamiento Auto-supervisado con Tareas Pretexto
**Archivo:** [`taller_clase_4/taller_clase_4.ipynb`](taller_clase_4/taller_clase_4.ipynb)

**Objetivo:** Implementar un modelo CNN o ViT con tareas de pretexto para entrenamiento auto-supervisado en ImageNet.

**Características principales:**
- Implementación de Vision Transformer (ViT)
- Tarea de pretexto: Jigsaw Puzzle
- Entrenamiento en ImageNet-mini
- Comparación entre CNN y arquitecturas Transformer

**Tecnologías:** PyTorch, Vision Transformers, KaggleHub

---

### 📦 Taller 5: Detección de Objetos con Bounding Boxes
**Archivos:**
- [`taller_clase_5/taller_clase_5_efficientnet.ipynb`](taller_clase_5/taller_clase_5_efficientnet.ipynb) (EfficientNet)
- [`taller_clase_5/taller_clase_5_vanilla_cnn.ipynb`](taller_clase_5/taller_clase_5_vanilla_cnn.ipynb) (CNN Vanilla)

**Objetivo:** Implementar sistemas de detección de objetos que predigan coordenadas de bounding boxes.

**Características principales:**
- **EfficientNet:** Backbone pre-entrenado congelado + cabeza de regresión
- **CNN Vanilla:** Arquitectura simple desde cero
- Dataset Caltech-101 con anotaciones de bounding boxes
- Evaluación con métricas de IoU y precisión de localización

**Tecnologías:** PyTorch, EfficientNet, OpenCV

---

### ✂️ Taller 6: Segmentación de Imágenes con SAM
**Archivo:** [`taller_clase_6/taller_clase_6_segmentacion_predictor.ipynb`](taller_clase_6/taller_clase_6_segmentacion_predictor.ipynb)

**Objetivo:** Implementar un sistema de segmentación de imágenes usando SAM (Segment Anything Model) para generar anotaciones de segmentación.

**Características principales:**
- Uso del modelo SAM pre-entrenado (ViT-H)
- Segmentación de objetos usando prompts de bounding boxes
- Generación de anotaciones de segmentación para el dataset Caltech 256
- Trabajo con la clase "airplanes-101" como caso de estudio
- Exportación de máscaras en formato COCO

**Tecnologías:** SAM (Segment Anything Model), PyTorch, OpenCV, pycocotools

---

## 🛠️ Tecnologías Utilizadas

- **Machine Learning:** PyTorch, scikit-learn, AutoGluon
- **Visión por Computadora:** OpenCV, PIL, scikit-image
- **Modelos Pre-entrenados:** ResNet50, MobileNet, EfficientNet, CLIP, ViT, SAM
- **Visualización:** Matplotlib, Seaborn, Plotly
- **Interfaz de Usuario:** Streamlit
- **Procesamiento de Datos:** Pandas, NumPy
- **Búsqueda Vectorial:** FAISS
- **Segmentación:** SAM (Segment Anything Model), pycocotools

## 📊 Datasets Utilizados

1. **Mango Leaf Disease** - Clasificación de enfermedades en hojas
2. **LabelMe-12-50k** - Clasificación de imágenes generales
3. **Caltech 256** - Recuperación de imágenes y segmentación
4. **ImageNet-mini** - Entrenamiento auto-supervisado
5. **Caltech-101** - Detección de objetos

## 🎯 Objetivos de Aprendizaje

Cada taller está diseñado para desarrollar habilidades específicas en:

- **Taller 1:** Extracción de características tradicionales y clasificación
- **Taller 2:** Transfer learning y modelos pre-entrenados
- **Taller 3:** Sistemas multimodales y recuperación de información
- **Taller 4:** Entrenamiento auto-supervisado y arquitecturas modernas
- **Taller 5:** Detección de objetos y regresión espacial
- **Taller 6:** Segmentación de imágenes y modelos foundation como SAM

## 📝 Notas Importantes

- Cada taller incluye conceptos teóricos explicados en archivos `conceptos_taller_X.txt`
- Los notebooks están completamente documentados con explicaciones paso a paso
- Se incluyen visualizaciones y análisis de resultados en cada taller
- Los modelos entrenados se guardan para reutilización posterior

## 🤝 Contribuciones

Este repositorio es parte de un curso académico. Para sugerencias o mejoras, por favor contacta al instructor del curso.

## 📄 Licencia

Este proyecto es para fines educativos únicamente.
