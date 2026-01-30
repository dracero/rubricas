"""
============================================================================
SISTEMA COLABA - EVALUADOR MULTI-AGENTE (VERSIÓN QDRANT)
============================================================================
Evaluador de Apuntes de Cátedra usando Rúbricas y Contexto Vectorial en Qdrant.
"""

import os
import json
import re
import time
import threading
import logging
from typing import Dict, List, Any, Optional

# Intentar importar pypdf
try:
    import pypdf
except ImportError:
    print("⚠️  Librería 'pypdf' no encontrada. Ejecuta: pip install pypdf")

# Google ADK y Generative AI
from google.adk.agents import Agent
from google import genai
from google.genai import types

# Persistencia y Embeddings
from sentence_transformers import SentenceTransformer
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:
    print("⚠️  Librería 'qdrant-client' no encontrada.")

# Para Colab Secrets
try:
    from google.colab import userdata
    USING_COLAB = True
except ImportError:
    USING_COLAB = False

# LangSmith para observabilidad
try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Decorador dummy si no está disponible
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator
    print("⚠️ LangSmith SDK no instalado. Ejecuta: pip install langsmith")


# ============================================================================
# CONFIGURACIÓN LANGSMITH
# ============================================================================

def setup_langsmith():
    """Configurar LangSmith para trazabilidad completa"""
    if not LANGSMITH_AVAILABLE:
        print("⚠️ LangSmith no disponible")
        return False
        
    try:
        api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
        if not api_key and USING_COLAB:
            try:
                api_key = userdata.get("LANGSMITH_API_KEY") or userdata.get("LANGCHAIN_API_KEY")
            except:
                pass
        
        if not api_key:
            print("⚠️ LangSmith: No API Key found.")
            return False

        langsmith_config = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": api_key,
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "rubricador_qdrant_evaluator"
        }

        for key, value in langsmith_config.items():
            if value:
                os.environ[key] = value

        print("✅ LangSmith configurado exitosamente (Proyecto: rubricador_qdrant_evaluator)")
        return True
    except Exception as e:
        print(f"⚠️ Error configurando LangSmith: {e}")
        return False

# ============================================================================
# UTILIDADES (Rate Limiter y Retry)
# ============================================================================

class GlobalRateLimiter:
    """Evita errores 429 en el Free Tier de Gemini"""
    _instance = None
    _lock = threading.Lock()
    _last_call = 0
    _min_interval = 2.0  # Ajustado para evitar bloqueos

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalRateLimiter, cls).__new__(cls)
            return cls._instance

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

rate_limiter = GlobalRateLimiter()

def llamar_llm_con_retry(func, max_intentos: int = 4, backoff_base: int = 10):
    for intento in range(max_intentos):
        rate_limiter.wait()
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if intento < max_intentos - 1:
                    wait_time = backoff_base * (2 ** intento)
                    print(f"⏳ Rate limit. Esperando {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None
            else:
                raise e
    return None

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class ConfiguracionColaba:
    def __init__(self):
        # Intentar cargar desde entorno o colab
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.QDRANT_URL = os.getenv('QDRANT_URL')
        self.QDRANT_KEY = os.getenv('QDRANT_KEY')

        if USING_COLAB:
            try:
                if not self.GOOGLE_API_KEY: self.GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
                if not self.QDRANT_URL: self.QDRANT_URL = userdata.get('QDRANT_URL')
                if not self.QDRANT_KEY: self.QDRANT_KEY = userdata.get('QDRANT_KEY')
            except:
                pass

# ============================================================================
# AGENTE 1: CONTEXTO (Qdrant)
# ============================================================================

class AgenteContexto(Agent):
    """Carga rúbrica y consulta Qdrant para contexto normativo"""
    
    def __init__(self, config):
        super().__init__(name="AgenteContexto")
        self._config = config
        self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._collection_name = "rubricas_entidades"
        
        # Inicializar Qdrant
        if config.QDRANT_URL:
            self._client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_KEY
            )
        else:
            self._client = None
            print("⚠️ Sin conexión a Qdrant (Solo modo local)")

    @traceable(name="AgenteContexto.obtener_conocimiento", run_type="retriever")
    def obtener_conocimiento(self, rubrica_path: str) -> Dict:
        """Obtiene conocimiento de la rúbrica y contexto RAG de Qdrant"""
        print(f"📚 [Agente Contexto] Procesando rúbrica: {rubrica_path}")
        
        # Metadata para LangSmith
        qdrant_metadata = {
            "rubrica_path": rubrica_path,
            "qdrant_hits_count": 0,
            "qdrant_similarity_scores": [],
            "retrieved_context_length": 0
        }
        
        try:
            with open(rubrica_path, "r", encoding="utf-8") as f:
                texto_rubrica = f.read()
        except FileNotFoundError:
            return {"rubrica_original": "", "contexto_qdrant": "", "_langsmith_metadata": qdrant_metadata}

        contexto_normativo = []
        similarity_scores = []
        
        # RAG con Qdrant: Buscar conceptos clave de la rúbrica en la normativa indexada
        if self._client:
            try:
                # Usamos el inicio de la rúbrica como query vectorial simple
                query_vec = self._embedding_model.encode(texto_rubrica[:500]).tolist()
                
                # Usar query_points (API moderna de qdrant-client 1.7+)
                hits = []
                try:
                    result = self._client.query_points(
                        collection_name=self._collection_name,
                        query=query_vec,
                        limit=10
                    )
                    hits = result.points if hasattr(result, 'points') else result
                except Exception as e:
                    print(f"⚠️ Error en búsqueda Qdrant: {e}")
                    hits = []

                for hit in hits:
                    payload = hit.payload
                    score = getattr(hit, 'score', 0.0)
                    similarity_scores.append(score)
                    contexto_normativo.append(f"- [{score:.3f}] {payload.get('nombre')}: {payload.get('contexto', '')[:300]}")
                
                # Actualizar metadata para LangSmith
                qdrant_metadata["qdrant_hits_count"] = len(hits)
                qdrant_metadata["qdrant_similarity_scores"] = similarity_scores
                qdrant_metadata["avg_similarity_score"] = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                    
            except Exception as e:
                print(f"⚠️ Error consultando Qdrant: {e}")

        contexto_str = "\n".join(contexto_normativo)
        qdrant_metadata["retrieved_context_length"] = len(contexto_str)
        
        print(f"   📊 Qdrant: {qdrant_metadata['qdrant_hits_count']} hits, avg_score: {qdrant_metadata.get('avg_similarity_score', 0):.3f}")

        return {
            "rubrica_original": texto_rubrica,
            "contexto_qdrant": contexto_str,
            "_langsmith_metadata": qdrant_metadata
        }

# ============================================================================
# AGENTE 2: DOCUMENTO (PDF Input)
# ============================================================================

class AgenteDocumento(Agent):
    """Extrae texto de PDF"""
    def __init__(self, config):
        super().__init__(name="AgenteDocumento")

    @traceable(name="AgenteDocumento.extraer_texto_pdf", run_type="parser")
    def extraer_texto_pdf(self, pdf_path: str) -> str:
        """Extrae texto de un archivo PDF para evaluación"""
        print(f"📄 [Agente Documento] Extrayendo texto de: {pdf_path}")
        try:
            reader = pypdf.PdfReader(pdf_path)
            texto = ""
            num_pages = len(reader.pages)
            for page in reader.pages:
                texto += page.extract_text() + "\n"
            
            resultado = texto.strip()
            print(f"   📊 PDF: {num_pages} páginas, {len(resultado)} caracteres extraídos")
            return resultado
        except Exception as e:
            print(f"❌ Error leyendo PDF: {e}")
            return ""

# ============================================================================
# AGENTE 3: AUDITOR (Evaluator)
# ============================================================================

class AgenteAuditor(Agent):
    """Evalúa el documento contra la rúbrica"""
    def __init__(self, config):
        super().__init__(name="AgenteAuditor")
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    @traceable(name="AgenteAuditor.evaluar_documento", run_type="llm")
    def evaluar_documento(self, conocimiento: Dict, documento_texto: str) -> str:
        """Evalúa el documento usando la rúbrica y contexto RAG"""
        print("⚖️ [Agente Auditor] Evaluando documento con Gemini...")
        
        # Extraer metadata de Qdrant si está disponible
        qdrant_meta = conocimiento.get('_langsmith_metadata', {})
        
        prompt = f"""
        Eres un AUDITOR ACADÉMICO riguroso.
        
        OBJETIVO:
        Evaluar los APUNTES DE CÁTEDRA proporcionados usando la RÚBRICA y el CONTEXTO NORMATIVO.
        
        DOCUMENTO A EVALUAR (Fragmento):
        {documento_texto[:15000]}
        
        CONTEXTO NORMATIVO (Qdrant - {qdrant_meta.get('qdrant_hits_count', 0)} documentos recuperados):
        {conocimiento['contexto_qdrant']}
        
        RÚBRICA DE REFERENCIA:
        {conocimiento['rubrica_original']}
        
        INSTRUCCIONES:
        1. Para cada criterio de la rúbrica, asigna una calificación y JUSTIFICA con evidencia del texto.
        2. Sé crítico: Si falta bibliografía o los links están rotos (simulado), señálalo.
        3. Genera un informe detallado que sirva de feedback al profesor autor de los apuntes.
        4. Usa formato Markdown claro.
        
        INFORME DE EVALUACIÓN:
        """
        
        # Estimación de tokens de entrada (aprox 4 chars = 1 token)
        input_chars = len(prompt)
        estimated_input_tokens = input_chars // 4
        print(f"   📊 Prompt: ~{estimated_input_tokens:,} tokens estimados")

        def hacer_llamada():
            return self._client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2, 
                    max_output_tokens=60000
                )
            )

        resp = llamar_llm_con_retry(hacer_llamada)
        
        if resp:
            output_text = resp.text
            estimated_output_tokens = len(output_text) // 4
            print(f"   📊 Respuesta: ~{estimated_output_tokens:,} tokens estimados")
            print(f"   📊 Qdrant context usado: {qdrant_meta.get('retrieved_context_length', 0)} chars, avg_score: {qdrant_meta.get('avg_similarity_score', 0):.3f}")
            return output_text
        return "Error en la generación."

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("SISTEMA COLABA - EVALUADOR DE APUNTES (BACKEND QDRANT)")
    print(f"{'='*80}\n")
    
    # Configurar LangSmith para observabilidad
    setup_langsmith()

    config = ConfiguracionColaba()
    if not config.GOOGLE_API_KEY:
        print("❌ Faltan credenciales (GOOGLE_API_KEY).")
        return

    # Rutas por defecto
    default_rubrica = "rubrica_calidad_apuntes_qdrant.txt"
    
    rubrica_path = input(f"Archivo de rúbrica [{default_rubrica}]: ") or default_rubrica
    pdf_path = input("Archivo PDF de apuntes a evaluar: ").strip()

    if not os.path.exists(rubrica_path):
        print(f"❌ Rúbrica no encontrada: {rubrica_path}")
        return
    if not os.path.exists(pdf_path):
        print(f"❌ PDF no encontrado: {pdf_path}")
        return

    # Inicializar
    agente_con = AgenteContexto(config)
    agente_doc = AgenteDocumento(config)
    agente_aud = AgenteAuditor(config)

    # Ejecutar
    conocimiento = agente_con.obtener_conocimiento(rubrica_path)
    texto_pdf = agente_doc.extraer_texto_pdf(pdf_path)
    
    if len(texto_pdf) < 50:
        print("⚠️ Advertencia: El PDF parece vacío o es una imagen escaneada.")

    forme_evaluacion = agente_aud.evaluar_documento(conocimiento, texto_pdf)

    # Guardar
    nombre_salida = f"evaluacion_{os.path.basename(pdf_path).replace('.pdf', '')}.md"
    with open(nombre_salida, "w", encoding="utf-8") as f:
        f.write(forme_evaluacion)

    print(f"\n✅ Evaluación completada. Guardada en: {nombre_salida}")

if __name__ == "__main__":
    main()
