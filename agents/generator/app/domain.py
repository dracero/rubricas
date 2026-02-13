"""
Domain Logic for Rubric Generator Agent.
Contains data structures, constants, and utility functions.

ADK agent classes are in adk_agents.py.
"""

import json
import re
import logging
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# UTILAJE: RATE LIMITER Y CACHE
# ============================================================================

class GlobalRateLimiter:
    _instance = None
    _lock = threading.Lock()
    _last_call = 0
    _min_interval = 1.0  # Configurable
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

    # Remover bloques de código markdown si existen (incluyendo variantes como ``` json)
    texto = re.sub(r'^```[a-zA-Z]*\s*', '', texto.strip())
    texto = re.sub(r'```\s*$', '', texto)

    # Encontrar el JSON (buscar desde { hasta el último })
    inicio = texto.find('{')
    fin = texto.rfind('}')
    if inicio != -1 and fin != -1 and fin > inicio:
        texto = texto[inicio:fin + 1]

    # Remover comas trailing antes de } o ]
    texto = re.sub(r',\s*}', '}', texto)
    texto = re.sub(r',\s*]', ']', texto)

    # Escapar saltos de línea dentro de strings JSON
    texto = texto.replace('\r\n', '\\n').replace('\r', '\\n')

    # Reemplazar tabs por espacios
    texto = texto.replace('\t', ' ')

    # Remover caracteres de control problemáticos
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
        logger.warning(f"⚠️ JSON inválido después de limpieza: {e}")

    # 3. Fallback: extraer entidades y relaciones con regex
    logger.info("🔧 Intentando extracción con regex como fallback...")
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

    return resultado


def llamar_llm_con_retry(func, prompt_for_cache=None, max_intentos=3):
    if prompt_for_cache:
        cached = llm_cache.get(prompt_for_cache)
        if cached:
            return cached

    for i in range(max_intentos):
        try:
            rate_limiter.wait()
            res = func()
            if prompt_for_cache:
                llm_cache.set(prompt_for_cache, res)
            return res
        except Exception as e:
            if i == max_intentos - 1:
                raise e
            time.sleep(2 ** i)


# ============================================================================
# CONSTANTES
# ============================================================================

IEEE_LOM_CONTEXTS = {
    "school": "Educación escolar (primaria/secundaria)",
    "higher education": "Educación superior universitaria",
    "training": "Formación profesional/capacitación",
    "other": "Otro contexto educativo"
}

NIVELES_ESTUDIANTE = {
    "inicial": {
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


# ============================================================================
# ESTRUCTURAS DE DATOS
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
