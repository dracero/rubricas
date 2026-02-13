"""
Shared Configuration and Utilities for Rubricas System.
Provides environment loading, configuration classes, and observability setup.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env from project root (assuming this file is in rubricas/common/config.py)
# We need to find the root "rubricas" directory or the valid .env
# The original script loaded .env from its own directory (root of rubricas).
# So we should look for .env in the parent of 'common' (which is 'rubricas')
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback to current directory or standard locations
    load_dotenv()

# Force tracing early if key is present
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"

# LangSmith con OpenTelemetry
try:
    from langsmith.integrations.otel import configure as configure_langsmith_otel
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    configure_langsmith_otel = None
    print("⚠️ LangSmith SDK no instalado. Ejecuta: uv add langsmith>=0.4.26")

# Decorador traceable (fallback)
try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def get_current_run_tree():
        return None

def get_env_var(key: str, default: Any = None) -> Any:
    """
    Obtiene una variable de entorno.
    """
    return os.environ.get(key, default)

def setup_langsmith():
    """Configurar LangSmith con OpenTelemetry para ADK"""
    if not LANGSMITH_AVAILABLE:
        return False
        
    try:
        # Obtener configuración
        api_key = get_env_var("LANGSMITH_API_KEY")
        project_name = get_env_var("LANGSMITH_PROJECT", "rubricas_qdrant_system")
        
        # Diagnósticos
        print("\n🔍 LangSmith Diagnostics:")
        print(f"   - API Key found: {'Yes (starts with ' + api_key[:4] + '...)' if api_key else 'No'}")
        
        if not api_key:
            print("⚠️ LangSmith: No API Key found in .env file.")
            return False

        # Configurar variables críticas
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project_name
        os.environ["LANGSMITH_TRACING"] = "true"  # Forzar tracing explícito
        
        print(f"   - Project: {project_name}")
        print(f"   - Tracing Value: {os.environ.get('LANGSMITH_TRACING')}")

        # Configurar OpenTelemetry
        if configure_langsmith_otel:
            configure_langsmith_otel(project_name=project_name)
        
        print(f"✅ LangSmith configurado con OpenTelemetry (Proyecto: {project_name})")
        print(f"✅ Tracing Active: {os.environ.get('LANGSMITH_TRACING')}")
        return True
    except Exception as e:
        print(f"⚠️ Error configurando LangSmith: {e}")
        import traceback
        traceback.print_exc()
        return False

class ConfiguracionColaba:
    def __init__(self):
        # Cargar desde variables de entorno (.env)
        self.GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY")
        self.QDRANT_URL = get_env_var("QDRANT_URL")
        self.QDRANT_API_KEY = get_env_var("QDRANT_API_KEY") or get_env_var("QDRANT_KEY")
        self.QDRANT_KEY = self.QDRANT_API_KEY # Alias for backward compatibility
        
        # Modelo de Embeddings
        self.EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        
        # Validación
        if not self.GOOGLE_API_KEY:
            raise ValueError("❌ Falta GOOGLE_API_KEY. Verifique su archivo .env")
        if not self.QDRANT_URL:
            # Check for memory mode implied by local URL or empty
            pass
