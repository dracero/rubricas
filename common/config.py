"""
Shared Configuration and Utilities for Rubricas System.
Provides environment loading, configuration classes, and observability setup.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional
from functools import wraps
import time

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
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    configure_langsmith_otel = None
    Client = None
    print("⚠️ LangSmith SDK no instalado. Ejecuta: uv add langsmith>=0.4.26")

# Decorador traceable (fallback)
try:
    from langsmith import traceable as _base_traceable
    from langsmith.run_helpers import get_current_run_tree, traceable as langsmith_traceable
    
    # Enhanced traceable decorator with automatic metadata capture
    def traceable(name: str = None, run_type: str = "chain", **kwargs):
        """Enhanced traceable decorator that captures detailed execution metadata."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **wrapper_kwargs):
                # Capture execution metadata
                start_time = time.time()
                metadata = {
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "run_type": run_type,
                }
                
                # Add arguments info (sanitized)
                if args:
                    metadata["args_count"] = len(args)
                if wrapper_kwargs:
                    metadata["kwargs_keys"] = list(wrapper_kwargs.keys())
                
                # Call the original traceable decorator
                traced_func = _base_traceable(
                    name=name or func.__name__,
                    run_type=run_type,
                    metadata=metadata,
                    **kwargs
                )(func)
                
                try:
                    result = traced_func(*args, **wrapper_kwargs)
                    
                    # Add execution time to current run
                    execution_time = time.time() - start_time
                    run_tree = get_current_run_tree()
                    if run_tree:
                        run_tree.extra = run_tree.extra or {}
                        run_tree.extra["execution_time_seconds"] = execution_time
                    
                    return result
                except Exception as e:
                    # Log error in trace
                    run_tree = get_current_run_tree()
                    if run_tree:
                        run_tree.error = str(e)
                    raise
            
            return wrapper
        return decorator
    
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
    """Configurar LangSmith con OpenTelemetry para ADK con trazabilidad completa"""
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

        # Configurar variables críticas para máxima trazabilidad
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project_name
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        
        # Habilitar captura de tokens y metadata
        os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"  # Sincrónico para capturar todo
        os.environ["LANGCHAIN_VERBOSE"] = "true"
        
        print(f"   - Project: {project_name}")
        print(f"   - Tracing Value: {os.environ.get('LANGSMITH_TRACING')}")
        print(f"   - Verbose Mode: Enabled")

        # Configurar OpenTelemetry con instrumentación completa
        if configure_langsmith_otel:
            configure_langsmith_otel(project_name=project_name)
        
        # Inicializar cliente LangSmith para logging adicional
        if Client:
            client = Client(api_key=api_key)
            print(f"   - LangSmith Client initialized")
        
        print(f"✅ LangSmith configurado con trazabilidad completa")
        print(f"   📊 Capturando: Tokens, Latencia, Interacciones Qdrant, Trayectoria de Agentes")
        print(f"   🔗 Dashboard: https://smith.langchain.com/o/projects/p/{project_name}")
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
