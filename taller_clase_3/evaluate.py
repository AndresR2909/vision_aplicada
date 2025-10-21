from __future__ import annotations

import random

import numpy as np


def evaluate_text_search_accuracy(retrieval_system, test_queries, k=5):
    """
    Evaluar la precisión de búsqueda por texto

    Args:
        test_queries: Lista de tuplas (query_text, expected_category)
        k: Número de resultados a considerar

    Returns:
        Diccionario con métricas de evaluación
    """
    print('Evaluando precisión de búsqueda por texto...')

    correct_predictions = 0
    total_predictions = 0
    results_by_query = []

    for query_text, expected_category in test_queries:
        try:
            results = retrieval_system.search_by_text(query_text, k=k)

            # Verificar si la categoría esperada está en los resultados
            found_correct = any(
                result['category_name'] == expected_category
                for result in results
            )

            if found_correct:
                correct_predictions += 1

            total_predictions += 1

            # Guardar resultados para análisis detallado
            results_by_query.append({
                'query': query_text,
                'expected': expected_category,
                'found_correct': found_correct,
                'top_result': results[0]['category_name'] if results else None,
                'top_score': results[0]['score'] if results else 0,
            })

        except Exception as e:
            print(f"Error procesando consulta '{query_text}': {e}")
            continue

    accuracy = (
        correct_predictions / total_predictions
        if total_predictions > 0 else 0
    )

    print('\nResultados de evaluación:')
    print(f"  - Consultas procesadas: {total_predictions}")
    print(f"  - Predicciones correctas: {correct_predictions}")
    print(f"  - Precisión: {accuracy:.3f}")

    return {
        'accuracy': accuracy,
        'total_queries': total_predictions,
        'correct_predictions': correct_predictions,
        'detailed_results': results_by_query,
    }


def evaluate_image_search_consistency(
    retrieval_system, dataset, num_tests=10, k=5,
):
    """
    Evaluar la consistencia de búsqueda por imagen

    Args:
        num_tests: Número de pruebas a realizar
        k: Número de resultados a considerar

    Returns:
        Diccionario con métricas de evaluación
    """
    print(
        f"Evaluando consistencia de búsqueda por imagen "
        f"({num_tests} pruebas)...",
    )

    consistency_scores = []
    category_matches = []

    for i in range(num_tests):
        # Seleccionar imagen aleatoria
        query_idx = random.randint(0, len(dataset) - 1)
        query_image = dataset[query_idx]['image']
        query_category = dataset[query_idx]['category_name']

        try:
            results = retrieval_system.search_by_image(query_image, k=k)

            # Calcular consistencia
            # (cuántos resultados son de la misma categoría)
            same_category_count = sum(
                1 for result in results
                if result['category_name'] == query_category
            )
            consistency = same_category_count / k

            consistency_scores.append(consistency)
            category_matches.append(same_category_count)

        except Exception as e:
            print(f"Error en prueba {i+1}: {e}")
            continue

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    avg_matches = np.mean(category_matches) if category_matches else 0

    print('\nResultados de evaluación:')
    print(f"  - Pruebas realizadas: {len(consistency_scores)}")
    print(f"  - Consistencia promedio: {avg_consistency:.3f}")
    print(f"  - Coincidencias promedio por consulta: {avg_matches:.2f}")

    return {
        'avg_consistency': avg_consistency,
        'avg_matches': avg_matches,
        'num_tests': len(consistency_scores),
        'consistency_scores': consistency_scores,
    }
