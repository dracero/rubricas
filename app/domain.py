"""
Domain Logic for Rubric Generator.
Contains data structures, constants, and utility functions.

Moved from agents/generator/app/domain.py to be shared by the unified app.
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
# UTILITIES: RATE LIMITER & CACHE
# ============================================================================

class GlobalRateLimiter:
    _instance = None
    _lock = threading.Lock()
    _last_call = 0
    _min_interval = 1.0
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
    """Clean a JSON response that may have common formatting errors."""
    if not texto:
        return "{}"

    texto = re.sub(r'^```[a-zA-Z]*\s*', '', texto.strip())
    texto = re.sub(r'```\s*$', '', texto)

    inicio = texto.find('{')
    fin = texto.rfind('}')
    if inicio != -1 and fin != -1 and fin > inicio:
        texto = texto[inicio:fin + 1]

    texto = re.sub(r',\s*}', '}', texto)
    texto = re.sub(r',\s*]', ']', texto)

    texto = texto.replace('\r\n', '\\n').replace('\r', '\\n')
    texto = texto.replace('\t', ' ')
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', texto)

    return texto


def parsear_json_con_fallback(texto: str) -> dict:
    """Parse JSON with multiple fallback strategies."""
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    texto_limpio = limpiar_json_respuesta(texto)
    try:
        return json.loads(texto_limpio)
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON inválido después de limpieza: {e}")

    logger.info("🔧 Intentando extracción con regex como fallback...")
    resultado = {"entidades": [], "relaciones": []}

    entidad_pattern = r'"nombre"\s*:\s*"([^"]+)"\s*,\s*"tipo"\s*:\s*"([^"]+)"'
    for match in re.finditer(entidad_pattern, texto_limpio):
        resultado["entidades"].append({
            "nombre": match.group(1),
            "tipo": match.group(2),
            "contexto": "",
            "propiedades": {}
        })

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
# CONSTANTS
# ============================================================================

COMPLIANCE_CONTEXTS = {
    "operacional": "Cumplimiento operacional interno",
    "técnico": "Normativas técnicas o estándares industriales",
    "legal": "Marco legal, regulatorio y contractual",
    "estatutario": "Estatutos y reglamentos institucionales"
}

NIVELES_EXIGENCIA = {
    "inicial": {
        "nombre": "Cumplimiento Operacional (Básico)",
        "max_criterios": 6,
        "lenguaje": "directo, enfocado en procesos inmediatos",
        "ejemplos_requeridos": True,
        "descripcion": "Rúbrica para verificación rápida de procesos operativos"
    },
    "avanzado": {
        "nombre": "Nivel Técnico/Regulatorio (Intermedio)",
        "max_criterios": 12,
        "lenguaje": "técnico preciso, alineado con estándares",
        "ejemplos_requeridos": True,
        "descripcion": "Rúbrica para auditoría de cumplimiento técnico o normativo"
    },
    "critico": {
        "nombre": "Auditoría de Alta Criticidad (Legal/Estratégico)",
        "max_criterios": 20,
        "lenguaje": "formal, legalmente vinculante y riguroso",
        "ejemplos_requeridos": False,
        "descripcion": "Rúbrica exhaustiva para cumplimiento legal o de alta seguridad"
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Entidad:
    """Represents an entity in the ontology."""
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
    """Represents a relation between entities."""
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
    """Complete ontology structure."""
    entidades: List[Entidad]
    relaciones: List[Relacion]
    metadata: Dict[str, Any]
