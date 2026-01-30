"""
============================================================================
SISTEMA COLABA - EVALUADOR MULTI-AGENTE (VERSIÓN COLAB + PDF)
Soporta: PDF interactivo, Rate Limiting Global y Salida en Formato Rúbrica
============================================================================
"""

import os
import json
import re
import time
import threading
import logging
from typing import Dict, List, Any, Optional

# Intentar importar pypdf (debe estar disponible en el entorno Colab o instalarse con !pip)
try:
    import pypdf
except ImportError:
    print("⚠️  Librería 'pypdf' no encontrada. Se recomienda ejecutar: !pip install pypdf")

# Google ADK y Generative AI
from google.adk.agents import Agent
from google import genai
from google.genai import types

# Persistencia y Embeddings
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Suprimir warnings de Neo4j
logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)

# ============================================================================
# UTILIDADES DE CONTROL (Rate Limiting)
# ============================================================================

class GlobalRateLimiter:
    """Evita errores 429 en el Free Tier de Gemini (15 RPM)"""
    _instance = None
    _lock = threading.Lock()
    _last_call = 0
    _min_interval = 7.5  # Margen de seguridad para ~8 RPM

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
                wait_time = self._min_interval - elapsed
                if wait_time > 0.5:
                    print(f"⏳ [RateLimit] Esperando {wait_time:.1f}s para cumplir cuota...")
                time.sleep(wait_time)
            self._last_call = time.time()

rate_limiter = GlobalRateLimiter()

def llamar_llm_con_retry(func, max_intentos: int = 4, backoff_base: int = 20):
    for intento in range(max_intentos):
        rate_limiter.wait()
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if intento < max_intentos - 1:
                    # Extraer tiempo sugerido
                    retry_match = re.search(r'after (\d+(?:\.\d+)?)s', error_str.lower())
                    if not retry_match:
                        retry_match = re.search(r'in (\d+(?:\.\d+)?)s', error_str.lower())
                    
                    wait_time = float(retry_match.group(1)) + 2.0 if retry_match else backoff_base * (2 ** intento)
                    print(f"⏳ Rate limit alcanzado. Intento {intento+1}. Esperando {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    return None
            else:
                raise e
    return None

# ============================================================================
# CONFIGURACIÓN (Compatible con Colab)
# ============================================================================

class ConfiguracionColaba:
    def __init__(self):
        # Intentar Colab userdata primero
        try:
            from google.colab import userdata
            self.NEO4J_URI = userdata.get('NEO4J_URI')
            self.NEO4J_USERNAME = userdata.get('NEO4J_USERNAME')
            self.NEO4J_PASSWORD = userdata.get('NEO4J_PASSWORD')
            self.GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
        except:
            self.NEO4J_URI = os.getenv('NEO4J_URI')
            self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
            self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
            self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# ============================================================================
# AGENTE 1: CONTEXTO (Knowledge Agent)
# ============================================================================

class AgenteContexto(Agent):
    """Carga rúbrica y consulta Neo4j"""
    def __init__(self, config):
        super().__init__(name="AgenteContexto")
        self._config = config
        self._driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
        self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def obtener_conocimiento(self, rubrica_path: str) -> Dict:
        print(f"📚 [Agente Contexto] Procesando rúbrica: {rubrica_path}")
        with open(rubrica_path, "r", encoding="utf-8") as f:
            texto_rubrica = f.read()

        # Enriquecer con Neo4j (RAG)
        query_embedding = self._embedding_model.encode(texto_rubrica[:1000]).tolist()
        contexto_normativo = []
        with self._driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('entidad_embedding', 8, $emb)
                YIELD node, score RETURN node.nombre as n, node.contexto as c ORDER BY score DESC
            """, emb=query_embedding)
            for rec in result:
                contexto_normativo.append(f"- {rec['n']}: {rec['c'][:400]}")

        return {
            "rubrica_original": texto_rubrica,
            "contexto_neo4j": "\n".join(contexto_normativo)
        }

    def close(self):
        self._driver.close()

# ============================================================================
# AGENTE 2: DOCUMENTO (Input Agent)
# ============================================================================

class AgenteDocumento(Agent):
    """Extrae y estructura texto de PDF"""
    def __init__(self, config):
        super().__init__(name="AgenteDocumento")

    def extraer_texto_pdf(self, pdf_path: str) -> str:
        print(f"📄 [Agente Documento] Extrayendo texto de: {pdf_path}")
        try:
            reader = pypdf.PdfReader(pdf_path)
            texto = ""
            for page in reader.pages:
                texto += page.extract_text() + "\n"
            return texto.strip()
        except Exception as e:
            print(f"❌ Error leyendo PDF: {e}")
            return ""

# ============================================================================
# AGENTE 3: AUDITOR (Evaluator Agent)
# ============================================================================

class AgenteAuditor(Agent):
    """Genera la rúbrica completada"""
    def __init__(self, config):
        super().__init__(name="AgenteAuditor")
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def completar_rubrica(self, conocimiento: Dict, documento_texto: str) -> str:
        print("⚖️ [Agente Auditor] Aplicando rúbrica y generando documento final...")
        
        prompt = f"""
Eres un AUDITOR ACADÉMICO. Tu salida DEBE SER UNA COPIA EXACTA de la RÚBRICA ORIGINAL proporcionada, pero COMPLETADA con tu análisis.

DOCUMENTO A EVALUAR:
{documento_texto[:8000]}

CONTEXTO NORMATIVO (Neo4j):
{conocimiento['contexto_neo4j']}

RÚBRICA ORIGINAL A COMPLETAR:
{conocimiento['rubrica_original']}

REGLAS DE SALIDA:
1. Mantén la estructura, títulos y campos de la RÚBRICA ORIGINAL.
2. Para cada criterio, reemplaza '□ Cumple' o '□ No cumple' por '[X] Cumple' o '[X] No cumple' según tu evaluación.
3. Rellena el campo 'Observaciones' con una justificación técnica basada en el documento y las normativas.
4. Si hay niveles como 'CUMPLE TOTALMENTE/PARCIALMENTE', marca el correcto con [X].
5. Responde SOLO con el documento de la rúbrica completada.

RÚBRICA COMPLETADA:
"""

        def hacer_llamada():
            return self._client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=8192)
            )

        resp = llamar_llm_con_retry(hacer_llamada)
        return resp.text if resp else "Error en la generación."

# ============================================================================
# MAIN INTERACTIVO
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("SISTEMA COLABA - EVALUACIÓN DE GUÍAS DOCENTES")
    print(f"{'='*80}\n")

    config = ConfiguracionColaba()
    if not config.GOOGLE_API_KEY:
        print("❌ Error: No se encontró GOOGLE_API_KEY. Configúrala en Colab userdata.")
        return

    # 1. Solicitar rutas al usuario
    rubrica_path = input("Ingrese el nombre del archivo de rúbrica (.txt) [default: rubrica_igualdad_genero.txt]: ") or "rubrica_igualdad_genero.txt"
    pdf_path = input("Ingrese la ruta del archivo PDF a verificar: ").strip()

    if not os.path.exists(rubrica_path):
        print(f"❌ No se encontró la rúbrica en: {rubrica_path}")
        return
    if not os.path.exists(pdf_path):
        print(f"❌ No se encontró el PDF en: {pdf_path}")
        return

    # 2. Inicializar Agentes
    agente_con = AgenteContexto(config)
    agente_doc = AgenteDocumento(config)
    agente_aud = AgenteAuditor(config)

    try:
        # 3. Ejecutar flujo
        conocimiento = agente_con.obtener_conocimiento(rubrica_path)
        texto_pdf = agente_doc.extraer_texto_pdf(pdf_path)
        
        if not texto_pdf:
            print("❌ El PDF está vacío o no se pudo leer.")
            return

        rubrica_final = agente_aud.completar_rubrica(conocimiento, texto_pdf)

        # 4. Guardar resultado
        nombre_salida = f"evaluacion_{os.path.basename(pdf_path).replace('.pdf', '')}.txt"
        with open(nombre_salida, "w", encoding="utf-8") as f:
            f.write(rubrica_final)

        print(f"\n✅ Proceso completado exitosamente.")
        print(f"📄 Rúbrica completada guardada en: {nombre_salida}")
        print("\n" + "="*80)
        print("VISTA PREVIA DEL RESULTADO:")
        print("="*80)
        print(rubrica_final[:1500] + "...")

    finally:
        agente_con.close()

if __name__ == "__main__":
    main()
