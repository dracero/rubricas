"""
============================================================================
SISTEMA COLABA QDRANT - Generación de Rúbricas con Vector Search & LangSmith
============================================================================

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
    print("⚠️ Qdrant Client no instalado. Ejecuta: pip install qdrant-client")

# LangSmith
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("⚠️ LangSmith SDK no instalado. Ejecuta: pip install langsmith")

# Para Colab Secrets
try:
    from google.colab import userdata
    USING_COLAB = True
except ImportError:
    USING_COLAB = False


# ============================================================================
# CONFIGURACIÓN LANGSMITH
# ============================================================================

def setup_langsmith():
    """Configurar LangSmith para trazabilidad"""
    if not LANGSMITH_AVAILABLE:
        return False
        
    try:
        # Intentar obtener API Key
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        if not api_key and USING_COLAB:
            try:
                api_key = userdata.get("LANGSMITH_API_KEY")
            except:
                pass
        
        if not api_key:
            print("⚠️ LangSmith: No API Key found.")
            return False

        langsmith_config = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": api_key,
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "rubricas_qdrant_system"
        }

        for key, value in langsmith_config.items():
            if value:
                os.environ[key] = value

        print("✅ LangSmith configurado exitosamente")
        return True
    except Exception as e:
        print(f"⚠️ Error configurando LangSmith: {e}")
        return False


# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================

class ConfiguracionColaba:
    def __init__(self):
        self.GOOGLE_API_KEY = self._get_secret("GOOGLE_API_KEY")
        self.QDRANT_URL = self._get_secret("QDRANT_URL")
        self.QDRANT_API_KEY = self._get_secret("QDRANT_KEY")
        
        # Modelo de Embeddings
        self.EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        
        # Validación
        if not self.GOOGLE_API_KEY:
            raise ValueError("Falta GOOGLE_API_KEY")
        if not self.QDRANT_URL:
            print("⚠️ Advertencia: Falta QDRANT_URL, se usará modo memoria si es posible o fallará.")

    def _get_secret(self, key: str) -> str:
        # 1. Intentar variable de entorno
        val = os.environ.get(key)
        if val: return val
        
        # 2. Si estamos en Colab, intentar userdata
        if USING_COLAB:
            try:
                return userdata.get(key)
            except:
                return None
        return None


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
# ============================================================================

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

    def buscar_similares(self, texto_consulta: str, limit: int = 5, score_threshold: float = 0.7) -> List[Dict]:
        """Busca entidades similares por vector"""
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
            for hit in hits:
                payload = hit.payload.copy() if hit.payload else {}
                payload['score'] = hit.score
                resultados.append(payload)
                
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
        self.token_limit = 60000

    def procesar_documento(self, texto: str) -> Ontologia:
        """Procesa un documento y extrae una ontología"""
        prompt = self._construir_prompt_extraccion(texto)
        
        def hacer_llamada():
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=self.token_limit,
                    response_mime_type="application/json"
                )
            )
            return response.text

        try:
            print("�� [Agente Ontólogo] Extrayendo entidades y relaciones...")
            resultado = llamar_llm_con_retry(hacer_llamada)
            
            # Parsear respuesta
            data = json.loads(resultado)
            
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
                
            return Ontologia(entidades=entidades, relaciones=relaciones, metadata={})
            
        except Exception as e:
            print(f"⚠️ Error en Agente Ontólogo: {e}")
            return Ontologia([], [], {})

    def _construir_prompt_extraccion(self, texto: str) -> str:
        return f"""
        Analiza el siguiente texto normativo y extrae una ONTOLOGÍA de conceptos educativos.
        
        TEXTO:
        {texto[:20000]}  # Limitar contexto si es necesario
        
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
        # Límite aumentado para documentos extensos
        self.max_tokens = 60000 

    def generar_rubrica(self, prompt_usuario: str, contexto_rag: Dict, nivel: str = "avanzado") -> str:
        """Genera la rúbrica final adaptada al nivel educativo"""
        
        # Obtener configuración del nivel
        config_nivel = NIVELES_ESTUDIANTE.get(nivel, NIVELES_ESTUDIANTE["avanzado"])
        
        # Formatear contexto de Qdrant
        contexto_str = ""
        for item in contexto_rag.get("resultados", []):
            contexto_str += f"- [{item['nombre']}]: {item['contexto']}\n"
            if 'relaciones_salientes' in item:
                for rel in item['relaciones_salientes']:
                    contexto_str += f"  -> {rel['tipo']} -> {rel['destino']}\n"
        
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
        
        CONTEXTO NORMATIVO (Base de Conocimiento):
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
        
        def hacer_llamada():
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_generacion,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            # Verificar truncamiento
            if response.candidates and response.candidates[0].finish_reason != "STOP":
                print(f"⚠️ Advertencia: Respuesta finalizada por {response.candidates[0].finish_reason}")
            
            return response.text

        try:
            print(f"✍️ [Agente Rubricador] Generando rúbrica para nivel: {config_nivel['nombre']}...")
            return llamar_llm_con_retry(hacer_llamada)
        except Exception as e:
            print(f"⚠️ Error generando rúbrica: {e}")
            return "Error en generación."

# ============================================================================
# AGENTE 3: BÚSQUEDA (ADAPTADO)
# ============================================================================

class AgenteBusqueda:
    """Coordina búsquedas en Qdrant"""
    
    def __init__(self, config: ConfiguracionColaba, persistencia: AgentePersistenciaQdrant):
        self.config = config
        self.persistencia = persistencia
        
    def procesar_prompt(self, prompt: str) -> Dict:
        print(f"🔎 [Agente Búsqueda] Buscando información para: '{prompt[:50]}...'")
        
        # Búsqueda semántica directa
        resultados = self.persistencia.buscar_similares(prompt, limit=15)
        
        return {
            "prompt": prompt,
            "resultados": resultados,
            "cantidad": len(resultados)
        }

# ============================================================================
# SISTEMA PRINCIPAL
# ============================================================================

class SistemaColabaQdrant:
    """Orquestador del sistema con Qdrant"""
    
    def __init__(self):
        print("🚀 Iniciando Sistema Colaba (Edición Qdrant)...")
        setup_langsmith()
        
        self.config = ConfiguracionColaba()
        self.agente_persistencia = AgentePersistenciaQdrant(self.config)
        self.agente_ontologo = AgenteOntologo(self.config)
        self.agente_rubricador = AgenteRubricador(self.config)
        self.agente_busqueda = AgenteBusqueda(self.config, self.agente_persistencia)

    def cargar_normativa(self, texto_normativa: str):
        """Procesa y guarda una normativa"""
        ontologia = self.agente_ontologo.procesar_documento(texto_normativa)
        if ontologia.entidades:
            self.agente_persistencia.guardar_ontologia(ontologia)
            print(f"✅ Normativa cargada: {len(ontologia.entidades)} entidades")
        else:
            print("⚠️ No se extrajeron entidades.")

    def generar_rubrica(self, prompt: str, archivo_salida: str = None, nivel: str = "avanzado") -> str:
        """Flujo completo de generación con nivel educativo"""
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
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Inicializar sistema
    colaba = SistemaColabaQdrant()

    # =========================================================================
    # METADATOS IEEE LOM PARA LA NORMATIVA (Basado en análisis de ontología)
    # =========================================================================
    
    metadatos_normativa_lom = {
        "general": {
            "identifier": {"catalog": "colaba-qdrant", "entry": "norm-apuntes-001"},
            "title": "Normativa de Calidad para la Elaboración de Apuntes de Cátedra",
            "language": "es",
            "description": "Criterios de evaluación para desarrollo de conceptos, referencias bibliográficas y recursos web en apuntes universitarios",
            "keyword": ["apuntes", "calidad", "evaluación", "bibliografía", "recursos web", "precisión conceptual"],
            "structure": "hierarchical",
            "aggregationLevel": "2"
        },
        "lifeCycle": {
            "version": "1.0",
            "status": "final",
            "contribute": [{"role": "author", "entity": "Sistema Colaba Qdrant", "date": "2026-01-29"}]
        },
        "educational": {
            "intendedEndUserRole": ["teacher", "author"],
            "context": ["higher education"],
            "learningResourceType": ["policy document", "evaluation rubric", "reference"],
            "typicalAgeRange": "18+",
            "semanticDensity": "high",
            "interactivityType": "expositive"
        },
        "rights": {
            "cost": "no",
            "copyrightAndOtherRestrictions": "yes",
            "description": "Uso institucional académico"
        },
        "relation": [
            {"kind": "isBasedOn", "resource": {"identifier": "IEEE_LOM_1484.12.1-2020"}}
        ],
        "classification": [
            {
                "purpose": "educational objective",
                "taxonPath": {
                    "source": "Normativa Interna",
                    "taxon": [
                        {"id": "art1", "entry": "Desarrollo de Conceptos"},
                        {"id": "art2", "entry": "Referencias Bibliográficas"},
                        {"id": "art3", "entry": "Recursos y Enlaces Web"}
                    ]
                }
            }
        ]
    }
    
    # Validar metadatos IEEE LOM
    es_valido, errores = validar_metadatos_lom(metadatos_normativa_lom)
    if es_valido:
        print("✅ Metadatos IEEE LOM válidos")
    else:
        print(f"⚠️ Errores en metadatos: {errores}")

    # 1. Definir Normativa de Calidad de Apuntes (con metadatos IEEE LOM)
    normativa_apuntes = f"""
    NORMATIVA DE CALIDAD PARA LA ELABORACIÓN DE APUNTES DE CÁTEDRA
    
    === METADATOS IEEE LOM ===
    Identificador: {metadatos_normativa_lom['general']['identifier']['entry']}
    Título: {metadatos_normativa_lom['general']['title']}
    Idioma: {metadatos_normativa_lom['general']['language']}
    Estructura: {metadatos_normativa_lom['general']['structure']}
    Contexto Educativo: {metadatos_normativa_lom['educational']['context']}
    Tipo de Recurso: {metadatos_normativa_lom['educational']['learningResourceType']}
    Densidad Semántica: {metadatos_normativa_lom['educational']['semanticDensity']}
    
    === REQUISITOS MÍNIMOS PARA APROBACIÓN ===
    Todo apunte debe cumplir con los siguientes requisitos mínimos observables:
    
    1. ESTRUCTURA VISIBLE:
       - Título del tema claramente identificado
       - Nombre del autor y fecha de elaboración
       - Índice o secciones numeradas (para documentos > 3 páginas)
       - Párrafos diferenciados con separación visual
    
    2. EXTENSIÓN MÍNIMA:
       - Al menos 1 página por unidad temática principal
       - Mínimo 500 palabras por concepto clave desarrollado
    
    3. FUENTES DOCUMENTADAS:
       - Mínimo 2 referencias bibliográficas por tema
       - Formato de citación consistente (APA, IEEE u otro)
       - Distinción clara entre citas textuales y paráfrasis
    
    4. CONTENIDO VERIFICABLE:
       - Sin errores conceptuales en definiciones clave
       - Terminología técnica usada correctamente
       - Al menos 1 ejemplo propio por concepto abstracto
    
    === CONTENIDO NORMATIVO ===

    ARTÍCULO 1: DESARROLLO DE CONCEPTOS
    Los apuntes deben presentar el contenido disciplinar con rigor académico y claridad expositiva.
    
    Criterios de evaluación con EVIDENCIAS OBSERVABLES:
    
    - Precisión conceptual: 
      EVIDENCIA: Las definiciones coinciden con las fuentes bibliográficas citadas.
      INDICADOR: 0 errores conceptuales graves en términos clave.
    
    - Profundidad del desarrollo:
      EVIDENCIA: Cada concepto incluye: definición + explicación + al menos 1 ejemplo.
      INDICADOR: Mínimo 3 niveles de detalle (qué es, cómo funciona, para qué sirve).
    
    - Secuenciación lógica:
      EVIDENCIA: Uso de conectores lógicos entre párrafos (por lo tanto, en consecuencia, etc.)
      INDICADOR: El lector puede seguir la argumentación sin saltos abruptos.
    
    - Elaboración personal:
      EVIDENCIA: Presencia de resúmenes, esquemas o diagramas propios del autor.
      INDICADOR: Al menos 1 elemento visual propio (tabla, diagrama, esquema) por tema.
      NOTA: "Elaboración personal" se mide por la presencia de síntesis y reformulación, 
            NO por el rendimiento posterior del estudiante.
    
    - Síntesis:
      EVIDENCIA: Inclusión de resumen o conclusión al final de cada sección.
      INDICADOR: Resumen de máximo 100 palabras por sección principal.

    ARTÍCULO 2: REFERENCIAS BIBLIOGRÁFICAS
    Todo material docente debe estar fundamentado en fuentes académicas reconocidas.
    
    Criterios de evaluación con EVIDENCIAS OBSERVABLES:
    
    - Citación correcta:
      EVIDENCIA: Todas las citas siguen un formato estándar consistente.
      INDICADOR: 100% de las citas con formato APA, IEEE o ISO 690.
    
    - Pertinencia temporal:
      EVIDENCIA: Fecha de publicación de las fuentes consultadas.
      INDICADOR: Al menos 50% de las referencias de los últimos 10 años.
    
    - Clasificación de fuentes:
      EVIDENCIA: Identificación explícita de bibliografía "básica" vs "complementaria".
      INDICADOR: Sección diferenciada o marcado visual de cada tipo.
    
    - Diversidad de fuentes:
      EVIDENCIA: Tipos de fuentes utilizadas (libros, artículos, recursos web).
      INDICADOR: Mínimo 2 tipos diferentes de fuentes.

    ARTÍCULO 3: RECURSOS Y ENLACES WEB
    Los recursos digitales complementarios deben enriquecer el aprendizaje.
    
    Criterios de evaluación con EVIDENCIAS OBSERVABLES:
    
    - Validez de enlaces:
      EVIDENCIA: Comprobación de que los enlaces funcionan (HTTP 200).
      INDICADOR: 100% de enlaces activos al momento de la entrega.
    
    - Descripción de recursos:
      EVIDENCIA: Cada enlace tiene descripción de 1-2 oraciones.
      INDICADOR: Ningún enlace "suelto" sin contexto explicativo.
    
    - Fuentes confiables:
      EVIDENCIA: Dominio del sitio web (edu, gov, org, instituciones reconocidas).
      INDICADOR: Al menos 70% de enlaces a fuentes institucionales o académicas.
    
    - Integración con contenido:
      EVIDENCIA: El recurso web está mencionado en el texto principal.
      INDICADOR: Cada enlace tiene una referencia explícita en el cuerpo del apunte.
    
    === ESCALA DE CALIFICACIÓN ===
    4 - EXCELENTE: Cumple todos los indicadores + aporta elementos adicionales de valor.
    3 - SATISFACTORIO: Cumple todos los requisitos mínimos e indicadores principales.
    2 - EN DESARROLLO: Cumple requisitos mínimos pero falla en 1-2 indicadores.
    1 - INSUFICIENTE: No cumple requisitos mínimos OR falla en 3+ indicadores.
    """

    # 2. Definir Estándar IEEE LOM (Estructura completa según IEEE 1484.12.1-2020)
    estandar_lom = f"""
    Estándar IEEE LOM (Learning Object Metadata) - IEEE 1484.12.1-2020
    
    Este estándar define metadatos para describir recursos educativos (objetos de aprendizaje).
    
    CATEGORÍAS DEL ESQUEMA IEEE LOM:
    
    1. GENERAL - Información general del recurso
       - Identificador (catálogo + entrada)
       - Título, idioma, descripción
       - Palabras clave
       - Estructura: {IEEE_LOM_SCHEMA['general']['structure']}
       - Nivel de agregación: {IEEE_LOM_SCHEMA['general']['aggregationLevel']}
    
    2. CICLO DE VIDA (LifeCycle)
       - Versión y estado
       - Estados válidos: {IEEE_LOM_SCHEMA['lifeCycle']['status']}
       - Contribuyentes (rol, entidad, fecha)
    
    3. EDUCATIVA (Educational) - Características pedagógicas
       - Roles de usuario: {list(IEEE_LOM_ROLES.keys())}
       - Contextos: {list(IEEE_LOM_CONTEXTS.keys())}
       - Tipos de recurso: {IEEE_LOM_RESOURCE_TYPES[:5]}...
       - Tipo de interactividad: {IEEE_LOM_SCHEMA['educational']['interactivityType']}
       - Densidad semántica: {IEEE_LOM_SEMANTIC_DENSITY}
       - Dificultad: {IEEE_LOM_SCHEMA['educational']['difficulty']}
    
    4. DERECHOS (Rights)
       - Costo: sí/no
       - Restricciones de copyright
       - Descripción de licencia
    
    5. RELACIÓN (Relation)
       - Tipos: isBasedOn, requires, references, isPartOf
       - Permite vincular recursos educativos relacionados
    
    6. CLASIFICACIÓN (Classification)
       - Propósito: disciplina, prerequisito, objetivo educativo
       - TaxonPath: sistema de clasificación jerárquico
    """

    print("\n📚 Cargando documentos normativos en Qdrant...")
    print(f"   📋 Ontología utilizada: IEEE LOM (IEEE 1484.12.1-2020)")
    print(f"   📝 Puntuación ontología: 4.25/5.00 (Ver SKILL.md para análisis completo)")
    colaba.cargar_normativa(normativa_apuntes)
    colaba.cargar_normativa(estandar_lom)

    # 3. Seleccionar nivel educativo (INTERACTIVO)
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
    
    # 4. Generar Rúbrica
    print("\n📋 Generando rúbrica de evaluación de APUNTES DE CÁTEDRA...")
    prompt_usuario = """
    Genera una rúbrica detallada para evaluar la CALIDAD DE APUNTES DE CÁTEDRA.
    Básate en la 'NORMATIVA DE CALIDAD PARA LA ELABORACIÓN DE APUNTES' y usa la estructura de metadatos de 'IEEE LOM' donde aplique.
    
    Asegúrate de incluir criterios específicos para:
    1. Desarrollo de Conceptos (Precisión, Profundidad)
    2. Bibliografía (Citación, Pertinencia)
    3. Links y Recursos Web (Validez, Calidad)
    """
    
    rubrica = colaba.generar_rubrica(
        prompt=prompt_usuario,
        archivo_salida="rubrica_calidad_apuntes_qdrant.txt",
        nivel=nivel_seleccionado
    )
    
    print("\n✅ Proceso finalizado.")

