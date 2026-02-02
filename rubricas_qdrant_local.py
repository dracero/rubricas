"""
============================================================================
SISTEMA COLABA QDRANT - Generación de Rúbricas con Vector Search & LangSmith
============================================================================

Versión Local (adaptada de Colab)
Sistema multi-agente para generar rúbricas académicas utilizando:
- Qdrant: Base de datos vectorial para persistencia y RAG
- LangSmith: Trazabilidad y observabilidad
- Google Gemini: LLM para razonamiento y generación
"""

import json
import re
import logging
import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from collections import defaultdict
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env from the same directory as this script
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

# Google Generative AI
from google import genai
from google.genai import types
from google.adk.agents import Agent

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ Qdrant Client no instalado. Ejecuta: uv add qdrant-client")

# PDF Reader
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("⚠️ pypdf no instalado. Ejecuta: uv add pypdf")

# LangSmith con OpenTelemetry (método correcto para ADK)
try:
    from langsmith.integrations.otel import configure as configure_langsmith_otel
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    configure_langsmith_otel = None
    print("⚠️ LangSmith SDK no instalado. Ejecuta: uv add langsmith>=0.4.26")

# Decorador traceable (fallback si no está disponible)
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


# ============================================================================
# UTILS: ENV (Local version using python-dotenv)
# ============================================================================

def get_env_var(key: str, default: Any = None) -> Any:
    """
    Obtiene una variable de entorno.
    Las variables se cargan desde el archivo .env al inicio del script.
    
    Args:
        key: Nombre de la variable de entorno
        default: Valor por defecto si no existe
    
    Returns:
        Valor de la variable o default
    """
    return os.environ.get(key, default)

# ============================================================================
# CONFIGURACIÓN LANGSMITH
# ============================================================================

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
        configure_langsmith_otel(project_name=project_name)
        
        print(f"✅ LangSmith configurado con OpenTelemetry (Proyecto: {project_name})")
        return True
    except Exception as e:
        print(f"⚠️ Error configurando LangSmith: {e}")
        return False


# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================

class ConfiguracionColaba:
    def __init__(self):
        # Cargar desde variables de entorno (.env)
        self.GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY")
        self.QDRANT_URL = get_env_var("QDRANT_URL")
        self.QDRANT_API_KEY = get_env_var("QDRANT_API_KEY") or get_env_var("QDRANT_KEY")
        
        # Modelo de Embeddings
        self.EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        
        # Validación
        if not self.GOOGLE_API_KEY:
            raise ValueError("❌ Falta GOOGLE_API_KEY. Verifique su archivo .env")
        if not self.QDRANT_URL:
            print("⚠️ Advertencia: Falta QDRANT_URL, se usará modo memoria si es posible o fallará.")


# ============================================================================
# ONTOLOGÍA IEEE LOM (IEEE 1484.12.1-2020) - Constantes y Esquema
# ============================================================================

# Roles de usuario según IEEE LOM Educational
IEEE_LOM_ROLES = {
    "teacher": "Docente/Profesor",
    "author": "Autor de contenido",
    "learner": "Estudiante/Aprendiz", 
    "manager": "Gestor/Administrador"
}

# Contextos educativos IEEE LOM
IEEE_LOM_CONTEXTS = {
    "school": "Educación escolar (primaria/secundaria)",
    "higher education": "Educación superior universitaria",
    "training": "Formación profesional/capacitación",
    "other": "Otro contexto educativo"
}

# Tipos de recursos de aprendizaje IEEE LOM
IEEE_LOM_RESOURCE_TYPES = [
    "exercise", "simulation", "questionnaire", "diagram", "figure",
    "graph", "index", "slide", "table", "narrative text", "exam",
    "experiment", "problem statement", "self assessment", "lecture",
    "policy document", "evaluation rubric", "reference"
]

# Niveles de densidad semántica IEEE LOM
IEEE_LOM_SEMANTIC_DENSITY = ["very low", "low", "medium", "high", "very high"]

# Estructura del esquema IEEE LOM para validación
IEEE_LOM_SCHEMA = {
    "general": {
        "identifier": {"catalog": str, "entry": str},
        "title": str,
        "language": str,
        "description": str,
        "keyword": list,
        "structure": ["hierarchical", "collection", "networked", "branched", "linear"],
        "aggregationLevel": ["1", "2", "3", "4"]
    },
    "lifeCycle": {
        "version": str,
        "status": ["draft", "final", "revised", "unavailable"],
        "contribute": list
    },
    "educational": {
        "interactivityType": ["active", "expositive", "mixed"],
        "learningResourceType": list,
        "interactivityLevel": IEEE_LOM_SEMANTIC_DENSITY,
        "semanticDensity": IEEE_LOM_SEMANTIC_DENSITY,
        "intendedEndUserRole": list,
        "context": list,
        "typicalAgeRange": str,
        "difficulty": ["very easy", "easy", "medium", "difficult", "very difficult"],
        "typicalLearningTime": str
    },
    "rights": {
        "cost": ["yes", "no"],
        "copyrightAndOtherRestrictions": ["yes", "no"],
        "description": str
    },
    "relation": list,
    "classification": list
}

# Niveles educativos para adaptación de rúbricas
NIVELES_ESTUDIANTE = {
    "primer_año": {
        "nombre": "Primer Año Universitario",
        "max_criterios": 5,
        "lenguaje": "simple y directo, evitando jerga técnica innecesaria",
        "ejemplos_requeridos": True,
        "descripcion": "Rúbrica simplificada con criterios básicos y claros"
    },
    "avanzado": {
        "nombre": "Estudiante Avanzado (3°-5° año)",
        "max_criterios": 12,
        "lenguaje": "técnico-académico apropiado para el nivel",
        "ejemplos_requeridos": True,
        "descripcion": "Rúbrica intermedia con criterios detallados"
    },
    "posgrado": {
        "nombre": "Posgrado/Investigación",
        "max_criterios": 20,
        "lenguaje": "especializado y preciso",
        "ejemplos_requeridos": False,
        "descripcion": "Rúbrica exhaustiva con todos los criterios"
    }
}


def validar_metadatos_lom(metadatos: Dict) -> Tuple[bool, List[str]]:
    """
    Valida que los metadatos cumplan con el esquema IEEE LOM.
    
    Args:
        metadatos: Diccionario con metadatos a validar
        
    Returns:
        Tuple con (es_valido, lista_de_errores)
    """
    errores = []
    
    # Campos obligatorios de General
    if "general" not in metadatos:
        errores.append("Falta categoría 'general' (obligatoria)")
    else:
        general = metadatos["general"]
        if not general.get("title"):
            errores.append("Falta 'general.title' (obligatorio)")
        if not general.get("description"):
            errores.append("Falta 'general.description' (obligatorio)")
        if not general.get("language"):
            errores.append("Falta 'general.language' (obligatorio)")
    
    # Validar Educational si existe
    if "educational" in metadatos:
        edu = metadatos["educational"]
        if edu.get("context"):
            contextos = edu["context"] if isinstance(edu["context"], list) else [edu["context"]]
            for ctx in contextos:
                if ctx not in IEEE_LOM_CONTEXTS:
                    errores.append(f"Contexto educativo '{ctx}' no válido. Use: {list(IEEE_LOM_CONTEXTS.keys())}")
    
    return len(errores) == 0, errores


# ============================================================================
# ESTRUCTURAS DE DATOS (Mantenidas de rubricas.py)
# ============================================== ==============================

@dataclass
class Entidad:
    """Representa una entidad en la ontología"""
    nombre: str
    tipo: str
    propiedades: Dict[str, Any]
    contexto: str
    embedding: Optional[List[float]] = None
    fecha_creacion: Optional[str] = None
    validada: bool = False
    
    def to_dict(self):
        return {
            "nombre": self.nombre,
            "tipo": self.tipo,
            "propiedades": self.propiedades,
            "contexto": self.contexto,
            "validada": self.validada,
            "fecha_creacion": self.fecha_creacion or datetime.now().isoformat()
        }

@dataclass
class Relacion:
    """Representa una relación entre entidades"""
    origen: str
    destino: str
    tipo: str
    propiedades: Dict[str, Any]
    confianza: float = 1.0
    
    def to_dict(self):
        return {
            "origen": self.origen,
            "destino": self.destino,
            "tipo": self.tipo,
            "propiedades": self.propiedades,
            "confianza": self.confianza
        }

@dataclass
class Ontologia:
    """Estructura completa de la ontología"""
    entidades: List[Entidad]
    relaciones: List[Relacion]
    metadata: Dict[str, Any]


# ============================================================================
# UTILAJE: RATE LIMITER Y CACHE (Reutilizados)
# ============================================================================

class GlobalRateLimiter:
    _instance = None
    _lock = threading.Lock()
    _last_call = 0
    _min_interval = 1.0 # Configurable
    _call_count = 0

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalRateLimiter, cls).__new__(cls)
            return cls._instance

    def wait(self):
        with self._lock:
            self._call_count += 1
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

rate_limiter = GlobalRateLimiter()

class LLMCache:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMCache, cls).__new__(cls)
                cls._instance._cache = {}
            return cls._instance
    
    def get(self, prompt):
        key = hashlib.md5(prompt[:1000].encode()).hexdigest()
        return self._cache.get(key)
        
    def set(self, prompt, response):
        key = hashlib.md5(prompt[:1000].encode()).hexdigest()
        self._cache[key] = response

llm_cache = LLMCache()

def limpiar_json_respuesta(texto: str) -> str:
    """
    Limpia una respuesta JSON que puede tener errores de formato comunes.
    Maneja: comas trailing, comillas no escapadas, saltos de línea en strings, etc.
    """
    if not texto:
        return "{}"
    
    # Remover bloques de código markdown si existen
    texto = re.sub(r'^```json\s*', '', texto.strip())
    texto = re.sub(r'^```\s*', '', texto)
    texto = re.sub(r'\s*```$', '', texto)
    
    # Encontrar el JSON (buscar desde { hasta el último })
    inicio = texto.find('{')
    fin = texto.rfind('}')
    if inicio != -1 and fin != -1 and fin > inicio:
        texto = texto[inicio:fin + 1]
    
    # Remover comas trailing antes de } o ]
    texto = re.sub(r',\s*}', '}', texto)
    texto = re.sub(r',\s*]', ']', texto)
    
    # Escapar saltos de línea dentro de strings JSON
    # Reemplazar newlines literales que no están escapados
    texto = texto.replace('\r\n', '\\n').replace('\r', '\\n')
    
    # Reemplazar tabs por espacios
    texto = texto.replace('\t', ' ')
    
    # Remover caracteres de control problemáticos (excepto \n y \t ya procesados)
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', texto)
    
    return texto


def parsear_json_con_fallback(texto: str) -> dict:
    """
    Intenta parsear JSON con múltiples estrategias de fallback.
    """
    # 1. Intentar parse directo
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    
    # 2. Limpiar y reintentar
    texto_limpio = limpiar_json_respuesta(texto)
    try:
        return json.loads(texto_limpio)
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON inválido después de limpieza: {e}")
        print(f"   📝 Fragmento problemático: ...{texto_limpio[max(0, e.pos-50):e.pos+50]}...")
    
    # 3. Fallback: extraer entidades y relaciones con regex
    print("   🔧 Intentando extracción con regex como fallback...")
    resultado = {"entidades": [], "relaciones": []}
    
    # Extraer entidades con regex
    entidad_pattern = r'"nombre"\s*:\s*"([^"]+)"\s*,\s*"tipo"\s*:\s*"([^"]+)"'
    for match in re.finditer(entidad_pattern, texto_limpio):
        resultado["entidades"].append({
            "nombre": match.group(1),
            "tipo": match.group(2),
            "contexto": "",
            "propiedades": {}
        })
    
    # Extraer relaciones con regex
    relacion_pattern = r'"origen"\s*:\s*"([^"]+)"\s*,\s*"destino"\s*:\s*"([^"]+)"\s*,\s*"tipo"\s*:\s*"([^"]+)"'
    for match in re.finditer(relacion_pattern, texto_limpio):
        resultado["relaciones"].append({
            "origen": match.group(1),
            "destino": match.group(2),
            "tipo": match.group(3),
            "propiedades": {}
        })
    
    if resultado["entidades"]:
        print(f"   ✅ Fallback exitoso: {len(resultado['entidades'])} entidades, {len(resultado['relaciones'])} relaciones")
    
    return resultado

def llamar_llm_con_retry(func, prompt_for_cache=None, max_intentos=3):
    if prompt_for_cache:
        cached = llm_cache.get(prompt_for_cache)
        if cached: return cached
        
    for i in range(max_intentos):
        try:
            rate_limiter.wait()
            res = func()
            if prompt_for_cache: llm_cache.set(prompt_for_cache, res)
            return res
        except Exception as e:
            if i == max_intentos - 1: raise e
            time.sleep(2 ** i)


# ============================================================================
# AGENTE PERSISTENCIA QDRANT
# ============================================================================

class AgentePersistenciaQdrant:
    """Gestiona la persistencia en Qdrant Vector DB"""
    
    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.collection_name = "rubricas_entidades"
        
        self._inicializar_coleccion()

    def _inicializar_coleccion(self):
        """Crea la colección si no existe"""
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                print(f"📦 Creando colección Qdrant: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"⚠️ Error inicializando Qdrant: {e}")

    def generar_embedding(self, texto: str) -> List[float]:
        return self.embedding_model.encode(texto).tolist()

    def guardar_ontologia(self, ontologia: Ontologia):
        """Guarda entidades y relaciones en Qdrant"""
        points = []
        
        # Mapear relaciones por entidad origen para guardarlas en payload
        relaciones_por_entidad = defaultdict(list)
        for rel in ontologia.relaciones:
            relaciones_por_entidad[rel.origen].append(rel.to_dict())
            
        for entidad in ontologia.entidades:
            # Generar ID determinista basado en nombre
            point_id = hashlib.md5(entidad.nombre.encode()).hexdigest()
            
            # Generar embedding del contexto + nombre
            texto_embedding = f"{entidad.nombre}: {entidad.contexto}"
            vector = self.generar_embedding(texto_embedding)
            
            # Construir payload
            payload = entidad.to_dict()
            payload["relaciones_salientes"] = relaciones_por_entidad[entidad.nombre]
            
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
            
        # Upsert en lotes
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"✅ Guardadas {len(points)} entidades en Qdrant")
                return True
            except Exception as e:
                print(f"❌ Error guardando en Qdrant: {e}")
                return False
        return False

    @traceable(name="AgentePersistenciaQdrant.buscar_similares", run_type="retriever")
    def buscar_similares(self, texto_consulta: str, limit: int = 5, score_threshold: float = 0.7) -> List[Dict]:
        """Busca entidades similares por vector (trazado via OpenTelemetry)"""
        vector = self.generar_embedding(texto_consulta)
        
        try:
            # Usar query_points (API moderna de qdrant-client 1.7+)
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Obtener puntos del resultado
            hits = result.points if hasattr(result, 'points') else result
            
            resultados = []
            scores = []
            for hit in hits:
                payload = hit.payload.copy() if hit.payload else {}
                score = hit.score
                payload['score'] = score
                scores.append(score)
                resultados.append(payload)
            
            # Log de métricas para observabilidad
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"   📊 Qdrant Search: {len(resultados)} hits, avg_score: {avg_score:.3f}, threshold: {score_threshold}")
                
            return resultados
        except Exception as e:
            print(f"⚠️ Error en búsqueda Qdrant: {e}")
            return []


# ============================================================================
# AGENTE 1: ONTÓLOGO (ADAPTADO QDRANT)
# ============================================================================

class AgenteOntologo:
    """Agente que extrae entidades y relaciones de textos normativos"""

    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.agent = Agent(
            name="ontologo",
            model="gemini-2.5-flash",
            instruction="Eres un experto en ontologías educativas. Extrae conceptos y relaciones.",
            description="Extrae entidades y relaciones"
        )
        self.token_limit = 60000  # Límite amplio para respuestas JSON completas

    @traceable(name="AgenteOntologo.procesar_documento", run_type="chain")
    def procesar_documento(self, texto: str) -> Ontologia:
        """Procesa un documento y extrae una ontología (trazado via OpenTelemetry)"""
        prompt = self._construir_prompt_extraccion(texto)
        
        # Estimar tokens de entrada
        input_chars = len(prompt)
        estimated_input_tokens = input_chars // 4
        print(f"   📊 Prompt ontología: ~{estimated_input_tokens:,} tokens estimados")
        
        token_usage = {}
        
        def hacer_llamada():
            nonlocal token_usage
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=self.token_limit,
                    response_mime_type="application/json"
                )
            )
            
            # Capturar tokens reales de Gemini
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return response.text

        try:
            print("🔬 [Agente Ontólogo] Extrayendo entidades y relaciones...")
            resultado = llamar_llm_con_retry(hacer_llamada)
            
            # Mostrar tokens reales y registrar en LangSmith
            if token_usage:
                print(f"   📊 Tokens Gemini: {token_usage.get('prompt_tokens', 0):,} in, {token_usage.get('completion_tokens', 0):,} out")
                
                # Registrar en LangSmith si está activo
                rt = get_current_run_tree()
                if rt:
                    rt.add_metadata({
                        "token_usage": token_usage,
                        "model": "gemini-2.5-flash"
                    })
            
            # Parsear respuesta JSON con fallback robusto
            data = parsear_json_con_fallback(resultado)
            
            entidades = []
            relaciones = []
            
            # Procesar entidades
            for e in data.get("entidades", []):
                entidades.append(Entidad(
                    nombre=e["nombre"],
                    tipo=e["tipo"],
                    propiedades=e.get("propiedades", {}),
                    contexto=e.get("contexto", ""),
                    fecha_creacion=datetime.now().isoformat()
                ))
                
            # Procesar relaciones
            for r in data.get("relaciones", []):
                relaciones.append(Relacion(
                    origen=r["origen"],
                    destino=r["destino"],
                    tipo=r["tipo"],
                    propiedades=r.get("propiedades", {}),
                    confianza=r.get("confianza", 1.0)
                ))
            
            print(f"   📊 Ontología extraída: {len(entidades)} entidades, {len(relaciones)} relaciones")
            return Ontologia(entidades=entidades, relaciones=relaciones, metadata=token_usage)
            
        except Exception as e:
            print(f"⚠️ Error en Agente Ontólogo: {e}")
            return Ontologia([], [], {})

    def _construir_prompt_extraccion(self, texto: str) -> str:
        return f"""
        Analiza el siguiente texto normativo y extrae una ONTOLOGÍA de conceptos educativos.
        
        TEXTO:
        {texto[:20000]}  # Límite de contexto
        
        INSTRUCCIONES:
        1. Identifica ENTIDADES clave: conceptos, criterios, niveles, requisitos.
        2. Identifica RELACIONES (MÍNIMO 3 por entidad):
           - REQUIERE, COMPLEMENTA, DEFINE, EJEMPLIFICA, PERTENECE_A, ES_PARTE_DE.
           - Busca relaciones explícitas e IMPLÍCITAS.
           - Conecta densamente los conceptos.
        3. Normaliza nombres (snake_case preferiblemente para IDs).
        
        Responde SOLO con JSON con estructura:
        {{
          "entidades": [
            {{ "nombre": "id_unico", "tipo": "concepto", "contexto": "definición breve", "propiedades": {{...}} }}
          ],
          "relaciones": [
            {{ "origen": "id_1", "destino": "id_2", "tipo": "REQUIERE" }}
          ]
        }}
        """

# ============================================================================
# AGENTE 2: RUBRICADOR (OUTPUT EXTENDED)
# ============================================================================

class AgenteRubricador:
    """Genera rúbricas usando RAG y Gemini"""
    
    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.agent = Agent(
            name="rubricador",
            model="gemini-2.5-flash", 
            instruction="Experto en evaluación educativa y diseño de rúbricas.",
            description="Genera rúbricas detalladas"
        )
        # Límite amplio para documentos extensos
        self.max_tokens = 60000 

    @traceable(name="AgenteRubricador.generar_rubrica", run_type="chain")
    def generar_rubrica(self, prompt_usuario: str, contexto_rag: Dict, nivel: str = "avanzado") -> str:
        """Genera la rúbrica final adaptada al nivel educativo con tracking completo"""
        
        # Obtener configuración del nivel
        config_nivel = NIVELES_ESTUDIANTE.get(nivel, NIVELES_ESTUDIANTE["avanzado"])
        
        # Formatear contexto de Qdrant con scores
        contexto_str = ""
        qdrant_scores = []
        for item in contexto_rag.get("resultados", []):
            score = item.get('score', 0)
            qdrant_scores.append(score)
            contexto_str += f"- [{score:.3f}] [{item.get('nombre', 'N/A')}]: {item.get('contexto', '')[:300]}\n"
            if 'relaciones_salientes' in item:
                for rel in item['relaciones_salientes']:
                    contexto_str += f"  -> {rel['tipo']} -> {rel['destino']}\n"
        
        avg_qdrant_score = sum(qdrant_scores) / len(qdrant_scores) if qdrant_scores else 0
        
        # Instrucciones adaptadas al nivel
        instrucciones_nivel = f"""
        ADAPTACIÓN AL NIVEL EDUCATIVO: {config_nivel['nombre']}
        - Máximo de criterios a incluir: {config_nivel['max_criterios']}
        - Estilo de lenguaje: {config_nivel['lenguaje']}
        - Incluir ejemplos concretos: {'SÍ, obligatorio' if config_nivel['ejemplos_requeridos'] else 'Opcional'}
        - Descripción: {config_nivel['descripcion']}
        """
        
        prompt_generacion = f"""
        Eres un ARQUITECTO PEDAGÓGICO experto en diseño de instrumentos de evaluación.
        
        SOLICITUD: {prompt_usuario}
        
        {instrucciones_nivel}
        
        CONTEXTO NORMATIVO (Base de Conocimiento - {len(qdrant_scores)} documentos, avg_score: {avg_qdrant_score:.3f}):
        {contexto_str}
        
        TAREA:
        Generar una RÚBRICA DE EVALUACIÓN adaptada al nivel indicado.
        
        ESTRUCTURA OBLIGATORIA:
        1. INFORMACIÓN GENERAL (Materia, Nivel, Objetivos)
        2. COMPETENCIAS A EVALUAR (Cognitivas, Procedimentales, Actitudinales)
        3. MATRIZ DE EVALUACIÓN (Dimensiones, Criterios, Escala 1-4, Evidencias observables)
        4. NIVELES DE DOMINIO con ejemplos específicos de qué constituye cada nivel
        5. RECOMENDACIONES AL ESTUDIANTE
        
        REGLAS CRÍTICAS:
        - NO uses términos vagos como "efectivo" o "adecuado" sin definirlos.
        - Cada criterio debe tener EVIDENCIAS OBSERVABLES (qué se puede ver/medir).
        - Incluye REQUISITOS MÍNIMOS concretos para aprobar.
        - Usa Markdown.
        - Respeta el límite de {config_nivel['max_criterios']} criterios principales.
        """
        
        # Estimar tokens de entrada
        input_chars = len(prompt_generacion)
        estimated_input_tokens = input_chars // 4
        print(f"   📊 Prompt rúbrica: ~{estimated_input_tokens:,} tokens, contexto RAG: {len(contexto_str)} chars")
        
        token_usage = {}
        
        @traceable(name="Gemini.generar_fragmento", run_type="llm")
        def _llamar_modelo_trazado(contenido_prompt: str) -> Any:
            """Llamada individual trazada para LangSmith"""
            return self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contenido_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=self.max_tokens,
                )
            )

        def hacer_llamada_con_continuacion():
            """Genera la rúbrica con continuación automática si se trunca"""
            nonlocal token_usage
            
            respuesta_completa = ""
            contexto_continuacion = prompt_generacion
            max_continuaciones = 5
            continuaciones = 0
            
            while continuaciones < max_continuaciones:
                # Usar la función trazada en lugar de llamar directo
                response = _llamar_modelo_trazado(contexto_continuacion)
                
                # Capturar tokens reales de Gemini
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    prev_tokens = token_usage.get('total_tokens', 0)
                    token_usage = {
                        "prompt_tokens": token_usage.get('prompt_tokens', 0) + getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": token_usage.get('completion_tokens', 0) + getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": prev_tokens + getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                    
                    # Registrar tokens para ESTA llamada específica en su propio trace
                    rt = get_current_run_tree()
                    if rt:
                        rt.add_metadata({
                            "token_usage_call": {
                                "prompt": getattr(response.usage_metadata, 'prompt_token_count', 0),
                                "completion": getattr(response.usage_metadata, 'candidates_token_count', 0)
                            }
                        })
                
                texto_parcial = response.text if response.text else ""
                respuesta_completa += texto_parcial
                
                # Verificar si la respuesta está completa
                finish_reason = None
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = response.candidates[0].finish_reason
                
                if finish_reason == "STOP" or finish_reason is None:
                    # Respuesta completa
                    break
                elif str(finish_reason) in ["MAX_TOKENS", "2", "FinishReason.MAX_TOKENS"]:
                    # Respuesta truncada, pedir continuación
                    continuaciones += 1
                    print(f"   ⚠️ Respuesta truncada (parte {continuaciones}), solicitando continuación...")
                    
                    # Nuevo prompt pidiendo continuar
                    contexto_continuacion = f"""
                    Continúa EXACTAMENTE donde quedaste. Esta es la continuación de una rúbrica que estabas generando.
                    
                    ÚLTIMO FRAGMENTO GENERADO (para contexto):
                    ...{texto_parcial[-500:]}
                    
                    INSTRUCCIÓN: Continúa desde ese punto. NO repitas lo anterior. Solo continúa la rúbrica.
                    """
                else:
                    print(f"   ⚠️ Respuesta finalizada por: {finish_reason}")
                    break
            
            if continuaciones > 0:
                print(f"   ✅ Rúbrica completada con {continuaciones} continuación(es)")
            
            return respuesta_completa

        try:
            print(f"✍️ [Agente Rubricador] Generando rúbrica para nivel: {config_nivel['nombre']}...")
            resultado = llamar_llm_con_retry(hacer_llamada_con_continuacion)
            
            if resultado:
                # Mostrar tokens reales y registrar en LangSmith
                if token_usage:
                    print(f"   📊 Tokens Gemini: {token_usage.get('prompt_tokens', 0):,} in, {token_usage.get('completion_tokens', 0):,} out")
                    
                    # Registrar en LangSmith si está activo
                    rt = get_current_run_tree()
                    if rt:
                        rt.add_metadata({
                            "token_usage": token_usage,
                            "model": "gemini-2.5-flash", 
                            "continuaciones": continuaciones if 'continuaciones' in locals() else 0
                        })
                else:
                    estimated_output_tokens = len(resultado) // 4
                    print(f"   📊 Respuesta rúbrica: ~{estimated_output_tokens:,} tokens (estimado)")
                
                print(f"   📊 Qdrant context: {len(qdrant_scores)} docs, avg_score: {avg_qdrant_score:.3f}")
                print(f"   📊 Longitud final: {len(resultado):,} caracteres")
            
            return resultado
        except Exception as e:
            print(f"⚠️ Error generando rúbrica: {e}")
            return "Error en generación."

# ============================================================================
# AGENTE 3: BÚSQUEDA (ADAPTADO)
# ============================================================================

class AgenteBusqueda:
    """Coordina búsquedas en Qdrant con tracking"""
    
    def __init__(self, config: ConfiguracionColaba, persistencia: AgentePersistenciaQdrant):
        self.config = config
        self.persistencia = persistencia
    
    @traceable(name="AgenteBusqueda.procesar_prompt", run_type="retriever")
    def procesar_prompt(self, prompt: str) -> Dict:
        """Procesa prompt y busca contexto en Qdrant con métricas"""
        print(f"🔎 [Agente Búsqueda] Buscando información para: '{prompt[:50]}...'")
        
        # Búsqueda semántica directa
        resultados = self.persistencia.buscar_similares(prompt, limit=15)
        
        # Extraer scores para métricas
        scores = [r.get('score', 0) for r in resultados]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"   📊 Búsqueda completada: {len(resultados)} resultados, avg_score: {avg_score:.3f}")
        
        return {
            "prompt": prompt,
            "resultados": resultados,
            "cantidad": len(resultados),
            "scores": scores,
            "avg_score": avg_score
        }

# ============================================================================
# FUNCIONES DE EXTRACCIÓN DE PDF
# ============================================================================

@traceable(name="extraer_texto_pdf", run_type="parser")
def extraer_texto_pdf(pdf_path: str) -> str:
    """
    Extrae texto de un archivo PDF normativo.
    
    Args:
        pdf_path: Ruta al archivo PDF
        
    Returns:
        Texto extraído del PDF
    """
    if not PYPDF_AVAILABLE:
        print("❌ pypdf no está disponible. Ejecuta: uv add pypdf")
        return ""
    
    print(f"📄 Extrayendo texto de: {pdf_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        texto = ""
        num_pages = len(reader.pages)
        for page in reader.pages:
            texto += page.extract_text() + "\n"
        
        resultado = texto.strip()
        print(f"   📊 PDF: {num_pages} páginas, {len(resultado):,} caracteres extraídos")
        return resultado
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {pdf_path}")
        return ""
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return ""


# ============================================================================
# SKILLS: ANÁLISIS DE ONTOLOGÍAS PARA DOCUMENTOS NORMATIVOS
# ============================================================================

# Palabras clave para clasificación automática de documentos
PALABRAS_CLAVE_ONTOLOGIA = {
    "IEEE_LOM": ["evaluación", "aprendizaje", "competencias", "objetivos", "didáctica", 
                 "pedagógico", "criterios", "rúbrica", "educativo", "estudiante", "docente"],
    "Dublin_Core": ["resolución", "expediente", "trámite", "procedimiento", "administrativo",
                    "documento", "publicación", "autor", "fecha"],
    "SCORM": ["SCORM", "LMS", "módulo", "interactivo", "tracking", "e-learning", "curso online"],
    "LRMI": ["licencia abierta", "reutilización", "compartir", "recurso abierto", "OER", "creative commons"]
}


def analizar_documento_normativo(texto: str) -> Dict[str, Any]:
    """
    Analiza las características de un documento normativo para determinar
    la ontología más apropiada.
    
    Skill basado en .agent/skills/SKILL.md
    
    Args:
        texto: Texto del documento normativo
        
    Returns:
        Dict con características detectadas
    """
    texto_lower = texto.lower()
    
    # Detectar tipo de documento
    tipo_documento = "general"
    if any(p in texto_lower for p in ["reglamento", "normativa", "ordenanza"]):
        tipo_documento = "reglamento"
    elif any(p in texto_lower for p in ["resolución", "decreto"]):
        tipo_documento = "resolución"
    elif any(p in texto_lower for p in ["guía", "manual", "instructivo"]):
        tipo_documento = "guía"
    
    # Detectar ámbito
    ambito = "general"
    if any(p in texto_lower for p in ["universidad", "académico", "educativo", "estudiante", "docente", "cátedra"]):
        ambito = "educativo"
    elif any(p in texto_lower for p in ["administrativo", "trámite", "expediente"]):
        ambito = "administrativo"
    
    # Detectar componentes pedagógicos
    tiene_componentes_pedagogicos = any(p in texto_lower for p in [
        "evaluación", "aprendizaje", "competencia", "criterio", "rúbrica",
        "calificación", "nivel educativo", "objetivos de aprendizaje"
    ])
    
    # Detectar si requiere interoperabilidad
    requiere_interoperabilidad = any(p in texto_lower for p in [
        "LMS", "moodle", "canvas", "sistema", "plataforma", "integración"
    ])
    
    # Detectar si es recurso abierto
    es_recurso_abierto = any(p in texto_lower for p in [
        "creative commons", "licencia abierta", "dominio público", "OER", "acceso abierto"
    ])
    
    # Detectar si requiere clasificación taxonómica
    requiere_clasificacion = any(p in texto_lower for p in [
        "artículo", "capítulo", "sección", "categoría", "clasificación"
    ])
    
    # Contar coincidencias de palabras clave por ontología
    conteo_palabras = {}
    for ontologia, palabras in PALABRAS_CLAVE_ONTOLOGIA.items():
        conteo = sum(1 for p in palabras if p.lower() in texto_lower)
        conteo_palabras[ontologia] = conteo
    
    return {
        "tipo_documento": tipo_documento,
        "ambito": ambito,
        "tiene_componentes_pedagogicos": tiene_componentes_pedagogicos,
        "requiere_interoperabilidad": requiere_interoperabilidad,
        "es_recurso_abierto": es_recurso_abierto,
        "requiere_clasificacion": requiere_clasificacion,
        "conteo_palabras_clave": conteo_palabras,
        "longitud_documento": len(texto)
    }


def calcular_ontologia_optima(caracteristicas: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
    """
    Calcula la ontología más apropiada basándose en las características del documento.
    
    Implementa el algoritmo de puntuación de SKILL.md
    
    Args:
        caracteristicas: Dict con características extraídas del documento
        
    Returns:
        Tuple con (nombre_ontologia, puntuacion, todas_las_puntuaciones)
    """
    pesos = {
        "pedagogico": 0.25,
        "simplicidad": 0.15,
        "interoperabilidad": 0.20,
        "derechos": 0.10,
        "taxonomia": 0.15,
        "lms": 0.15
    }
    
    # Puntuaciones base ajustadas según características
    puntuaciones_base = {
        "IEEE_LOM": {
            "pedagogico": 5 if caracteristicas.get("tiene_componentes_pedagogicos") else 2,
            "simplicidad": 3,
            "interoperabilidad": 4,
            "derechos": 4,
            "taxonomia": 5 if caracteristicas.get("requiere_clasificacion") else 3,
            "lms": 4
        },
        "Dublin_Core": {
            "pedagogico": 2,
            "simplicidad": 5,
            "interoperabilidad": 5,
            "derechos": 3,
            "taxonomia": 3,
            "lms": 2
        },
        "SCORM": {
            "pedagogico": 3,
            "simplicidad": 2,
            "interoperabilidad": 4,
            "derechos": 2,
            "taxonomia": 2,
            "lms": 5 if caracteristicas.get("requiere_interoperabilidad") else 3
        },
        "LRMI": {
            "pedagogico": 3,
            "simplicidad": 4,
            "interoperabilidad": 4,
            "derechos": 5 if caracteristicas.get("es_recurso_abierto") else 3,
            "taxonomia": 3,
            "lms": 3
        }
    }
    
    # Calcular puntuación ponderada
    resultados = {}
    for ontologia, scores in puntuaciones_base.items():
        # Bonus por coincidencia de palabras clave
        bonus = min(caracteristicas.get("conteo_palabras_clave", {}).get(ontologia, 0) * 0.1, 0.5)
        total = sum(scores[k] * pesos[k] for k in pesos) + bonus
        resultados[ontologia] = round(total, 2)
    
    mejor = max(resultados, key=resultados.get)
    return mejor, resultados[mejor], resultados


@traceable(name="ejecutar_skills_post_carga", run_type="chain")
def ejecutar_skills_post_carga(texto_documento: str, nombre_documento: str = "Documento") -> Dict[str, Any]:
    """
    Ejecuta los skills de análisis de ontología cuando se carga un documento normativo.
    
    Este es el punto de entrada principal del sistema de skills.
    
    Args:
        texto_documento: Texto del documento normativo
        nombre_documento: Nombre descriptivo del documento
        
    Returns:
        Dict con resultados del análisis y recomendación
    """
    print(f"\n{'='*60}")
    print(f"🔍 SKILL: Análisis de Ontología para: {nombre_documento}")
    print(f"{'='*60}")
    
    # Paso 1: Analizar características
    print("\n📋 Paso 1: Analizando características del documento...")
    caracteristicas = analizar_documento_normativo(texto_documento)
    
    print(f"   • Tipo: {caracteristicas['tipo_documento']}")
    print(f"   • Ámbito: {caracteristicas['ambito']}")
    print(f"   • Componentes pedagógicos: {'✅' if caracteristicas['tiene_componentes_pedagogicos'] else '❌'}")
    print(f"   • Requiere interoperabilidad: {'✅' if caracteristicas['requiere_interoperabilidad'] else '❌'}")
    print(f"   • Es recurso abierto: {'✅' if caracteristicas['es_recurso_abierto'] else '❌'}")
    
    # Paso 2: Calcular puntuaciones
    print("\n📊 Paso 2: Calculando puntuaciones de ontologías...")
    mejor_ontologia, puntuacion, todas_puntuaciones = calcular_ontologia_optima(caracteristicas)
    
    print("\n   Puntuaciones:")
    ranking = sorted(todas_puntuaciones.items(), key=lambda x: x[1], reverse=True)
    medallas = ["🥇", "🥈", "🥉", "  "]
    for i, (ont, score) in enumerate(ranking):
        marca = " ✅" if ont == mejor_ontologia else ""
        print(f"   {medallas[i]} {ont}: {score:.2f}/5.00{marca}")
    
    # Paso 3: Generar recomendación
    print(f"\n✅ RECOMENDACIÓN: {mejor_ontologia} (Puntuación: {puntuacion:.2f}/5.00)")
    
    # Justificación
    justificaciones = {
        "IEEE_LOM": "Ideal para documentos educativos con metadatos pedagógicos ricos.",
        "Dublin_Core": "Apropiado para documentos generales con metadatos básicos.",
        "SCORM": "Mejor para contenido e-learning interactivo empaquetado.",
        "LRMI": "Óptimo para recursos educativos abiertos con compatibilidad web."
    }
    print(f"   📝 {justificaciones.get(mejor_ontologia, 'Ontología seleccionada basada en análisis.')}")
    
    return {
        "documento": nombre_documento,
        "caracteristicas": caracteristicas,
        "ontologia_recomendada": mejor_ontologia,
        "puntuacion": puntuacion,
        "todas_puntuaciones": todas_puntuaciones,
        "justificacion": justificaciones.get(mejor_ontologia, "")
    }


# ============================================================================
# SISTEMA PRINCIPAL
# ============================================================================

class SistemaColabaQdrant:
    """Orquestador del sistema con Qdrant y LangSmith"""
    
    def __init__(self):
        print("🚀 Iniciando Sistema Colaba (Edición Local)...")
        self.langsmith_enabled = setup_langsmith()
        
        self.config = ConfiguracionColaba()
        self.agente_persistencia = AgentePersistenciaQdrant(self.config)
        self.agente_ontologo = AgenteOntologo(self.config)
        self.agente_rubricador = AgenteRubricador(self.config)
        self.agente_busqueda = AgenteBusqueda(self.config, self.agente_persistencia)

    @traceable(name="SistemaColaba.cargar_normativa", run_type="chain")
    def cargar_normativa(self, texto_normativa: str):
        """Procesa y guarda una normativa (trazado via OpenTelemetry)"""
        ontologia = self.agente_ontologo.procesar_documento(texto_normativa)
        if ontologia.entidades:
            self.agente_persistencia.guardar_ontologia(ontologia)
            print(f"✅ Normativa cargada: {len(ontologia.entidades)} entidades")
        else:
            print("⚠️ No se extrajeron entidades.")

    @traceable(name="SistemaColaba.generar_rubrica", run_type="chain")
    def generar_rubrica(self, prompt: str, archivo_salida: str = None, nivel: str = "avanzado") -> str:
        """Flujo completo de generación (trazado via OpenTelemetry)"""
        contexto = self.agente_busqueda.procesar_prompt(prompt)
        rubrica = self.agente_rubricador.generar_rubrica(prompt, contexto, nivel)
        
        if archivo_salida:
            with open(archivo_salida, 'w', encoding='utf-8') as f:
                f.write(rubrica)
                f.flush()
                os.fsync(f.fileno())
            size = os.path.getsize(archivo_salida)
            print(f"\n💾 Rúbrica guardada en: {archivo_salida} ({size/1024:.1f} KB)")
            
        return rubrica


# ============================================================================
# FUNCIÓN MAIN PARA EJECUCIÓN
# ============================================================================

# Documento normativo de fallback (se usa si no se proporciona PDF)
NORMATIVA_FALLBACK = """
NORMATIVA DE CALIDAD PARA LA ELABORACIÓN DE APUNTES DE CÁTEDRA

=== REQUISITOS MÍNIMOS PARA APROBACIÓN ===
Todo apunte debe cumplir con los siguientes requisitos mínimos observables:

1. ESTRUCTURA VISIBLE:
   - Título del tema claramente identificado
   - Nombre del autor y fecha de elaboración
   - Índice o secciones numeradas (para documentos > 3 páginas)

2. EXTENSIÓN MÍNIMA:
   - Al menos 1 página por unidad temática principal
   - Mínimo 500 palabras por concepto clave desarrollado

3. FUENTES DOCUMENTADAS:
   - Mínimo 2 referencias bibliográficas por tema
   - Formato de citación consistente (APA, IEEE u otro)

ARTÍCULO 1: DESARROLLO DE CONCEPTOS
- Precisión conceptual: Las definiciones coinciden con las fuentes bibliográficas citadas.
- Profundidad del desarrollo: Cada concepto incluye definición + explicación + ejemplo.

ARTÍCULO 2: REFERENCIAS BIBLIOGRÁFICAS
- Citación correcta: Todas las citas siguen un formato estándar consistente.
- Pertinencia temporal: Al menos 50% de las referencias de los últimos 10 años.

ARTÍCULO 3: RECURSOS Y ENLACES WEB
- Validez de enlaces: 100% de enlaces activos al momento de la entrega.
- Fuentes confiables: Al menos 70% de enlaces a fuentes institucionales o académicas.

=== ESCALA DE CALIFICACIÓN ===
4 - EXCELENTE: Cumple todos los indicadores + aporta elementos adicionales de valor.
3 - SATISFACTORIO: Cumple todos los requisitos mínimos e indicadores principales.
2 - EN DESARROLLO: Cumple requisitos mínimos pero falla en 1-2 indicadores.
1 - INSUFICIENTE: No cumple requisitos mínimos OR falla en 3+ indicadores.
"""


def main():
    """Punto de entrada principal para ejecución local"""
    
    print("\n" + "="*60)
    print("🚀 SISTEMA COLABA QDRANT - Generación de Rúbricas")
    print("="*60)
    
    # Inicializar sistema
    colaba = SistemaColabaQdrant()

    # =========================================================================
    # 1. SOLICITAR DOCUMENTO NORMATIVO (PDF o fallback)
    # =========================================================================
    print("\n" + "="*60)
    print("📄 CARGA DE DOCUMENTO NORMATIVO")
    print("="*60)
    print("\nIngrese la ruta del archivo PDF con el documento normativo.")
    print("(Presione Enter sin escribir nada para usar el documento de ejemplo)")
    print(f"Ejemplo: ./mi_normativa.pdf o /ruta/completa/documento.pdf")
    
    pdf_path = input("\n📁 Ruta del PDF normativo: ").strip()
    
    texto_normativa = ""
    nombre_documento = "Documento"
    
    if pdf_path:
        # Cargar desde PDF
        if os.path.exists(pdf_path):
            texto_normativa = extraer_texto_pdf(pdf_path)
            nombre_documento = os.path.basename(pdf_path)
            
            if not texto_normativa:
                print("⚠️ No se pudo extraer texto del PDF. Usando documento de ejemplo.")
                texto_normativa = NORMATIVA_FALLBACK
                nombre_documento = "Normativa de Ejemplo"
        else:
            print(f"❌ Archivo no encontrado: {pdf_path}")
            print("⚠️ Usando documento de ejemplo...")
            texto_normativa = NORMATIVA_FALLBACK
            nombre_documento = "Normativa de Ejemplo"
    else:
        print("ℹ️ Usando documento normativo de ejemplo...")
        texto_normativa = NORMATIVA_FALLBACK
        nombre_documento = "Normativa de Ejemplo"

    # =========================================================================
    # 2. EJECUTAR SKILLS DE ANÁLISIS DE ONTOLOGÍA
    # =========================================================================
    resultado_skills = ejecutar_skills_post_carga(texto_normativa, nombre_documento)
    
    # Usar la ontología recomendada
    ontologia_recomendada = resultado_skills.get("ontologia_recomendada", "IEEE_LOM")
    puntuacion_ontologia = resultado_skills.get("puntuacion", 0)

    # =========================================================================
    # 3. CARGAR NORMATIVA EN QDRANT
    # =========================================================================
    print("\n📚 Cargando documento normativo en Qdrant...")
    print(f"   📋 Ontología utilizada: {ontologia_recomendada}")
    print(f"   📝 Puntuación: {puntuacion_ontologia:.2f}/5.00")
    colaba.cargar_normativa(texto_normativa)
    
    # Cargar también el estándar de la ontología recomendada
    estandar_info = f"""
    Estándar de Ontología: {ontologia_recomendada}
    
    Este documento fue analizado y se determinó que la ontología {ontologia_recomendada}
    es la más apropiada para estructurar sus metadatos.
    
    Características del documento:
    - Tipo: {resultado_skills.get('caracteristicas', {}).get('tipo_documento', 'N/A')}
    - Ámbito: {resultado_skills.get('caracteristicas', {}).get('ambito', 'N/A')}
    - Componentes pedagógicos: {'Sí' if resultado_skills.get('caracteristicas', {}).get('tiene_componentes_pedagogicos') else 'No'}
    """
    colaba.cargar_normativa(estandar_info)

    # =========================================================================
    # 4. SELECCIONAR NIVEL EDUCATIVO (INTERACTIVO)
    # =========================================================================
    print("\n" + "="*60)
    print("📊 SELECCIÓN DE NIVEL EDUCATIVO")
    print("="*60)
    print("\nNiveles disponibles:")
    for key, val in NIVELES_ESTUDIANTE.items():
        print(f"  {key}: {val['nombre']} (máx. {val['max_criterios']} criterios)")
    
    print("\nOpciones rápidas: 1=primer_año, 2=avanzado, 3=posgrado")
    nivel_input = input("Nivel del estudiante [2=avanzado]: ").strip() or "2"
    
    nivel_map = {"1": "primer_año", "2": "avanzado", "3": "posgrado"}
    nivel_seleccionado = nivel_map.get(nivel_input, nivel_input)
    
    # Validar nivel
    if nivel_seleccionado not in NIVELES_ESTUDIANTE:
        print(f"⚠️ Nivel '{nivel_seleccionado}' no reconocido. Usando 'avanzado'.")
        nivel_seleccionado = "avanzado"
    
    print(f"\n✅ Nivel seleccionado: {NIVELES_ESTUDIANTE[nivel_seleccionado]['nombre']}")

    # =========================================================================
    # 5. GENERAR RÚBRICA
    # =========================================================================
    print("\n📋 Generando rúbrica de evaluación...")
    
    # Construir prompt basado en el documento cargado
    prompt_usuario = f"""
    Genera una rúbrica detallada para evaluar documentos según la normativa cargada: "{nombre_documento}".
    
    Usa la ontología {ontologia_recomendada} para estructurar los metadatos.
    
    Asegúrate de incluir:
    1. Criterios específicos basados en el contenido del documento normativo
    2. Evidencias observables para cada criterio
    3. Indicadores cuantificables
    4. Escala de calificación clara
    """
    
    # Definir nombre de archivo de salida
    nombre_salida = f"rubrica_{nombre_documento.replace('.pdf', '').replace(' ', '_')}.txt"
    
    rubrica = colaba.generar_rubrica(
        prompt=prompt_usuario,
        archivo_salida=nombre_salida,
        nivel=nivel_seleccionado
    )
    
    print("\n" + "="*60)
    print("✅ PROCESO FINALIZADO")
    print("="*60)
    print(f"\n📄 Documento normativo: {nombre_documento}")
    print(f"🔍 Ontología aplicada: {ontologia_recomendada} ({puntuacion_ontologia:.2f}/5.00)")
    print(f"📊 Nivel educativo: {NIVELES_ESTUDIANTE[nivel_seleccionado]['nombre']}")
    print(f"💾 Rúbrica guardada en: {nombre_salida}")


if __name__ == "__main__":
    main()

