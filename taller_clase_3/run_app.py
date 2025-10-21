#!/usr/bin/env python3
"""
Script para ejecutar la aplicación Streamlit con configuración optimizada
"""
from __future__ import annotations

import os
import subprocess
import sys


def setup_environment():
    """Configurar variables de entorno para evitar warnings"""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONWARNINGS'] = (
        'ignore::UserWarning:multiprocessing.resource_tracker'
    )


def main():
    """Ejecutar la aplicación Streamlit"""
    setup_environment()

    print('🚀 Iniciando aplicación de recuperación de imágenes...')
    print('📱 URL: http://localhost:8501')
    print('⚠️  Presiona Ctrl+C para detener la aplicación')

    try:
        # Ejecutar streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'ui_streamlit.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false',
        ])
    except KeyboardInterrupt:
        print('\n👋 Aplicación detenida por el usuario')
    except Exception as e:
        print(f"❌ Error ejecutando la aplicación: {e}")


if __name__ == '__main__':
    main()
