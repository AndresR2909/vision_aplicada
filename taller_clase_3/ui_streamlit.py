from __future__ import annotations

import multiprocessing
import os
import warnings

import numpy as np
import streamlit as st
from image_retrieval_system import CLIPImageRetrievalSystem

# Configurar tokenizers antes de importar otros módulos
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Configurar multiprocessing para macOS
multiprocessing.set_start_method('spawn', force=True)

# Suprimir warnings específicos de multiprocessing en macOS
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='multiprocessing.resource_tracker',
)
warnings.filterwarnings('ignore', category=FutureWarning)


# Configuración de la página
st.set_page_config(
    page_title='Sistema de Recuperación de Imágenes',
    page_icon='🔍',
    layout='wide',
)


@st.cache_resource
def load_retrieval_system():
    """Cargar el sistema de recuperación con cache"""
    filepath = 'index/caltech256_clip_index'

    # Verificar que los archivos del índice existen
    required_files = [
        f"{filepath}_image.index",
        f"{filepath}_text.index",
        f"{filepath}_image_embeddings.npy",
        f"{filepath}_text_embeddings.npy",
        f"{filepath}_metadata.pkl",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"Archivos del índice no encontrados: {missing_files}",
        )

    # Crear y cargar el sistema
    system = CLIPImageRetrievalSystem()
    system.load_index(filepath)

    # Verificar que los datos se cargaron correctamente
    if system.image_embeddings is None or system.text_embeddings is None:
        raise ValueError('Los embeddings no se cargaron correctamente')

    if not system.metadata:
        raise ValueError('Los metadatos no se cargaron correctamente')

    return system


def diagnose_system():
    """Diagnosticar problemas del sistema"""
    st.subheader('🔧 Diagnóstico del Sistema')

    filepath = 'index/caltech256_clip_index'
    required_files = [
        f"{filepath}_image.index",
        f"{filepath}_text.index",
        f"{filepath}_image_embeddings.npy",
        f"{filepath}_text_embeddings.npy",
        f"{filepath}_metadata.pkl",
    ]

    st.write('**Archivos del índice:**')
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            st.write(f"✅ {file} ({size:.1f} MB)")
        else:
            st.write(f"❌ {file} - NO ENCONTRADO")

    # Intentar cargar el sistema paso a paso
    st.write('\n**Prueba de carga:**')
    try:
        system = CLIPImageRetrievalSystem()
        st.write('✅ Sistema CLIP inicializado')

        system.load_index(filepath)
        st.write('✅ Índices cargados')

        if system.image_embeddings is not None:
            st.write(
                f"✅ Embeddings de imagen: {system.image_embeddings.shape}",
            )
        else:
            st.write('❌ Embeddings de imagen: None')

        if system.text_embeddings is not None:
            st.write(f"✅ Embeddings de texto: {system.text_embeddings.shape}")
        else:
            st.write('❌ Embeddings de texto: None')

        if system.metadata:
            st.write(f"✅ Metadatos: {len(system.metadata)} elementos")
        else:
            st.write('❌ Metadatos: Vacío')

    except Exception as e:
        st.error(f"❌ Error en diagnóstico: {e}")


def display_search_results(results, query_text=None, max_results=8):
    """Mostrar resultados de búsqueda en Streamlit"""
    if not results:
        st.warning('No se encontraron resultados')
        return

    st.subheader(f"Resultados para: '{query_text}'")
    st.write(f"Encontrados {len(results)} resultados")

    # Crear columnas para mostrar las imágenes
    num_cols = 4
    num_rows = (len(results) + num_cols - 1) // num_cols

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            result_idx = row * num_cols + col_idx
            if result_idx < len(results):
                result = results[result_idx]

                with cols[col_idx]:
                    # Mostrar imagen
                    st.image(
                        result['image'],
                        caption=(
                            f"#{result['rank']}: {result['category_name']}\n"
                            f"Score: {result['score']:.3f}"
                        ),
                        use_container_width=True,
                    )

                    # Información adicional
                    with st.expander('Detalles'):
                        st.write(f"**Categoría:** {result['category_name']}")
                        st.write(
                            f"**ID de categoría:** {result['category_id']}",
                        )
                        st.write(
                            f"**Score de similitud:** {result['score']:.4f}",
                        )
                        st.write(f"**Ranking:** {result['rank']}")


def main():
    """Función principal de la aplicación"""

    # Título y descripción
    st.title('🔍 Sistema de Recuperación de Imágenes')
    st.markdown("""
    **Sistema de búsqueda de imágenes por texto usando CLIP y FAISS**

    Este sistema permite buscar imágenes similares usando descripciones
    en texto natural.
    """)

    # Cargar el sistema de recuperación
    with st.spinner('Cargando sistema de recuperación...'):
        try:
            retrieval_system = load_retrieval_system()
            st.success('✅ Sistema cargado exitosamente')
        except Exception as e:
            st.error(f"❌ Error cargando el sistema: {str(e)}")
            st.error(
                '💡 Solución: Asegúrate de haber ejecutado el notebook '
                'para generar los índices',
            )

            # Mostrar diagnóstico si hay error
            if st.button('🔧 Ejecutar Diagnóstico'):
                diagnose_system()
            st.stop()

    # Información del sistema
    with st.expander('ℹ️ Información del Sistema'):
        st.write(
            f"**Total de imágenes indexadas:** "
            f"{len(retrieval_system.metadata)}",
        )

        # Verificar que los embeddings estén cargados correctamente
        if (
            retrieval_system.image_embeddings is not None and
            len(retrieval_system.image_embeddings.shape) > 1
        ):
            st.write(
                f"**Dimensiones de embeddings:** "
                f"{retrieval_system.image_embeddings.shape[1]}",
            )
        else:
            st.write('**Dimensiones de embeddings:** No disponibles')

        st.write(f"**Modelo CLIP:** {retrieval_system.model_id}")
        st.write(f"**Dispositivo:** {retrieval_system.device}")

    # Interfaz de búsqueda
    st.header('🔎 Búsqueda por Texto')

    # Input de texto
    query_text = st.text_input(
        'Ingresa tu consulta de búsqueda:',
        placeholder='Ej: dog, car, airplane, chair, etc.',
        help='Describe la imagen que quieres buscar',
    )

    # Número de resultados
    num_results = st.slider(
        'Número de resultados:',
        min_value=1,
        max_value=20,
        value=8,
        help='Selecciona cuántos resultados quieres ver',
    )

    # Botón de búsqueda
    if st.button('🔍 Buscar', type='primary'):
        if query_text.strip():
            with st.spinner('Buscando imágenes...'):
                try:
                    # Realizar búsqueda
                    results = retrieval_system.search_by_text(
                        query_text, k=num_results,
                    )

                    # Mostrar resultados
                    display_search_results(results, query_text, num_results)

                    # Estadísticas de búsqueda
                    if results:
                        scores = [r['score'] for r in results]
                        st.metric('Score promedio', f"{np.mean(scores):.3f}")
                        st.metric('Score máximo', f"{np.max(scores):.3f}")
                        st.metric('Score mínimo', f"{np.min(scores):.3f}")

                except Exception as e:
                    st.error(f"❌ Error en la búsqueda: {str(e)}")
        else:
            st.warning('⚠️ Por favor ingresa una consulta de búsqueda')

    # Ejemplos de consultas
    st.header('💡 Ejemplos de Consultas')

    example_queries = [
        'dog', 'cat', 'car', 'airplane', 'horse',
        'chair', 'table', 'bicycle', 'bird', 'flower',
        'red car', 'flying airplane', 'sitting dog',
        'wooden chair', 'beautiful flower',
    ]

    st.write('**Prueba con estos ejemplos:**')
    cols = st.columns(5)
    for i, example in enumerate(example_queries):
        with cols[i % 5]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.query_text = example
                st.rerun()

    # Mostrar consulta seleccionada
    if hasattr(st.session_state, 'query_text'):
        st.text_input(
            'Consulta seleccionada:',
            value=st.session_state.query_text, disabled=True,
        )

    # Información adicional
    st.header('📊 Estadísticas del Dataset')

    # Análisis de categorías
    category_counts = {}
    for metadata in retrieval_system.metadata:
        cat_name = metadata['category_name']
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    # Mostrar top categorías
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Top 10 Categorías')
        for cat, count in sorted_categories[:10]:
            st.write(f"• **{cat}**: {count} imágenes")

    with col2:
        st.subheader('Estadísticas Generales')
        st.metric('Total de categorías', len(category_counts))
        st.metric('Total de imágenes', len(retrieval_system.metadata))
        st.metric(
            'Promedio por categoría',
            f"{len(retrieval_system.metadata) / len(category_counts):.1f}",
        )

    # Footer
    st.markdown('---')
    st.markdown('**Sistema desarrollado con CLIP, FAISS y Streamlit**')


if __name__ == '__main__':
    main()
