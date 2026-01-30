"""
============================================================================
SISTEMA COLABA - Generación Automática de Rúbricas con ONTOLOGÍA DINÁMICA
Con Google ADK (Agents Development Kit) - VERSIÓN MEJORADA
============================================================================

Sistema multi-agente con ontología dinámica para generar rúbricas académicas:
- Validación ontológica con LLM
- Resolución automática de entidades duplicadas
- Schema auto-evolutivo
- Normalización semántica
- Métricas de consistencia ontológica

NUEVAS CARACTERÍSTICAS:
✓ AgenteValidadorOntologico - Valida y normaliza entidades
✓ OntologiaDinamica - Schema que evoluciona con el uso
✓ ResolucionEntidades - Detecta y fusiona duplicados
✓ ValidadorConsistencia - Métricas de calidad del grafo
✓ Schema-aware extraction - Extracción consciente del esquema existente

Autor: Sistema Colaba
Versión: 3.0 (Ontología Dinámica)
============================================================================
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import time
import threading

# Suprimir warnings de Neo4j (propiedades que no existen en base vacía)
logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)

# Neo4j
from neo4j import GraphDatabase

# Google ADK
from google.adk.agents import Agent

# Google Generative AI
from google import genai
from google.genai import types

# Sentence Transformers
from sentence_transformers import SentenceTransformer

import os

# Para Colab Secrets
try:
    from google.colab import userdata
    USING_COLAB = True
except ImportError:
    USING_COLAB = False


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
    entidad_canonica: Optional[str] = None  # Si es alias de otra entidad


@dataclass
class Relacion:
    """Representa una relación entre entidades"""
    origen: str
    destino: str
    tipo: str
    propiedades: Dict[str, Any]
    confianza: float = 1.0  # Score de confianza en la relación


@dataclass
class Ontologia:
    """Estructura completa de la ontología"""
    entidades: List[Entidad]
    relaciones: List[Relacion]
    metadata: Dict[str, Any]


@dataclass
class SchemaOntologico:
    """Define el schema dinámico de la ontología"""
    tipos_entidad: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tipos_relacion: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reglas_validacion: List[Dict[str, Any]] = field(default_factory=list)
    estadisticas: Dict[str, int] = field(default_factory=dict)


@dataclass
class ResultadoValidacion:
    """Resultado de validación ontológica"""
    entidad_validada: Entidad
    es_nueva: bool
    entidades_fusionadas: List[str] = field(default_factory=list)
    cambios_aplicados: List[str] = field(default_factory=list)
    score_confianza: float = 1.0
# ============================================================================
# CONFIGURACIÓN Y CREDENCIALES
# ============================================================================

class GlobalRateLimiter:
    """Control global de frecuencia para la API de Gemini (Free Tier: 15 RPM / 1M TPM)
       Para evitar errores 429 persistentes, limitamos a un máximo de 5 llamadas por minuto
       (una cada 12 segundos). Usamos Singleton para que afecte a todos los agentes.
    """
    _instance = None
    _lock = threading.Lock()
    _last_call = 0
    _min_interval = 12.0  # 5 llamadas por minuto máximo para mayor seguridad
    _call_count = 0  # Contador de llamadas para debugging

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalRateLimiter, cls).__new__(cls)
            return cls._instance

    def wait(self):
        """Espera el tiempo necesario para no exceder la cuota"""
        with self._lock:
            self._call_count += 1
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                # Solo imprimimos si la espera es significativa
                if wait_time > 1.0:
                    print(f"⏳ [RateLimit] Llamada #{self._call_count} - Esperando {wait_time:.1f}s para cumplir cuota API...")
                time.sleep(wait_time)
            self._last_call = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de uso del rate limiter"""
        return {
            'total_llamadas': self._call_count,
            'intervalo_minimo': self._min_interval,
            'ultima_llamada': self._last_call
        }

# Instancia global única
rate_limiter = GlobalRateLimiter()


# ============================================================================
# CACHÉ DE RESPUESTAS LLM
# ============================================================================

import hashlib

class LLMCache:
    """Caché simple para respuestas LLM - evita llamadas repetidas"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMCache, cls).__new__(cls)
                cls._instance._cache = {}
                cls._instance._hits = 0
                cls._instance._misses = 0
            return cls._instance
    
    def get_key(self, prompt: str) -> str:
        """Genera una clave hash para el prompt (primeros 1000 chars)"""
        return hashlib.md5(prompt[:1000].encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """Obtiene respuesta cacheada si existe"""
        key = self.get_key(prompt)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, prompt: str, response: str):
        """Guarda respuesta en caché"""
        key = self.get_key(prompt)
        with self._lock:
            # Limitar tamaño del caché a 100 entradas
            if len(self._cache) >= 100:
                # Eliminar la primera entrada (FIFO simple)
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[key] = response
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del caché"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self._cache)
        }
    
    def clear(self):
        """Limpia el caché"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

# Instancia global única del caché
llm_cache = LLMCache()

def llamar_llm_con_retry(func, prompt_for_cache: str = None, max_intentos: int = 4, backoff_base: int = 20):
    """
    Ejecuta una función con retry exponencial en caso de rate limit,
    respetando el limitador de frecuencia global y usando caché.
    
    Args:
        func: Función que hace la llamada LLM
        prompt_for_cache: Prompt opcional para cachear la respuesta
        max_intentos: Número máximo de reintentos
        backoff_base: Tiempo base para backoff exponencial (aumentado a 20s)
    """
    # 1. Verificar caché primero (evita llamada a API)
    if prompt_for_cache:
        cached = llm_cache.get(prompt_for_cache)
        if cached:
            print("📦 [Cache HIT] Usando respuesta cacheada")
            # Crear un objeto mock con atributo .text para compatibilidad
            class CachedResponse:
                def __init__(self, text):
                    self.text = text
            return CachedResponse(cached)
    
    # 2. Si no hay caché, hacer la llamada con retry
    for intento in range(max_intentos):
        # Primero respetar el limitador de frecuencia global
        rate_limiter.wait()

        try:
            result = func()
            
            # Guardar en caché si es exitoso y hay prompt
            if result and prompt_for_cache:
                response_text = result.text if hasattr(result, 'text') else str(result)
                llm_cache.set(prompt_for_cache, response_text)
            
            return result
        except Exception as e:
            error_str = str(e)

            # Detectar rate limit
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if intento < max_intentos - 1:
                    # Intentar extraer el tiempo sugerido del error
                    # Formatos comunes: "retry in X.Xs" o "retry after X.Xs"
                    wait_time = backoff_base * (2 ** intento)

                    retry_match = re.search(r'after (\d+(?:\.\d+)?)s', error_str.lower())
                    if not retry_match:
                        retry_match = re.search(r'in (\d+(?:\.\d+)?)s', error_str.lower())

                    if retry_match:
                        wait_time = float(retry_match.group(1)) + 2.0  # Margen extra aumentado

                    print(f"⏳ Rate limit alcanzado. Intento {intento+1}/{max_intentos}. Esperando {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Rate limit persistente después de {max_intentos} intentos")
                    return None
            else:
                # Si no es rate limit, mostrar y propagar para depuración si es crítico
                print(f"⚠️ Error inesperado en llamada LLM: {e}")
                raise

    return None


class ConfiguracionColaba:
    """Gestiona la configuración y credenciales del sistema"""

    def __init__(self):
        self.cargar_credenciales()

    def cargar_credenciales(self):
        """Carga credenciales desde Colab Secrets o variables de entorno"""
        if USING_COLAB:
            print("📱 Ejecutando en Google Colab - Cargando secrets...")
            try:
                self.NEO4J_URI = userdata.get('NEO4J_URI')
                self.NEO4J_USERNAME = userdata.get('NEO4J_USERNAME')
                self.NEO4J_PASSWORD = userdata.get('NEO4J_PASSWORD')
                self.GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
                print("✓ Credenciales cargadas desde Colab Secrets")
                print(f"  - Neo4j URI: {self.NEO4J_URI[:30]}...")
                print(f"  - Neo4j User: {self.NEO4J_USERNAME}")
            except Exception as e:
                raise ValueError(
                    f"❌ Error cargando secrets: {e}\n\n"
                    "Configura los secrets en Google Colab:\n"
                    "1. Haz clic en el icono 🔑 en el panel izquierdo\n"
                    "2. Añade: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY"
                )
        else:
            print("💻 Ejecutando localmente - Cargando variables de entorno...")
            self.NEO4J_URI = os.getenv('NEO4J_URI')
            self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
            self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
            self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            print("✓ Credenciales cargadas desde variables de entorno")

        # Validar
        if not all([self.NEO4J_URI, self.NEO4J_USERNAME,
                    self.NEO4J_PASSWORD, self.GOOGLE_API_KEY]):
            raise ValueError("❌ Faltan credenciales. Configúralas correctamente.")


# ============================================================================
# GESTOR DE ONTOLOGÍA DINÁMICA
# ============================================================================

class OntologiaDinamica:
    """Gestiona el schema ontológico que evoluciona con el uso"""

    def __init__(self, driver, client: genai.Client):
        self.driver = driver
        self.client = client
        self.schema_cache: Optional[SchemaOntologico] = None
        self.ultima_actualizacion: Optional[datetime] = None

    def cargar_schema_actual(self) -> SchemaOntologico:
        """Carga el schema actual desde Neo4j"""
        with self.driver.session() as session:
            # Obtener tipos de entidad y sus estadísticas
            tipos_entidad = session.run("""
                MATCH (e:Entidad)
                WITH e.tipo AS tipo,
                     count(*) AS cantidad,
                     collect(DISTINCT keys(e)) AS propiedades_usadas
                RETURN tipo, cantidad, propiedades_usadas
                ORDER BY cantidad DESC
            """).data()

            # Obtener tipos de relación
            tipos_relacion = session.run("""
                MATCH ()-[r]->()
                WITH type(r) AS tipo, count(*) AS cantidad
                RETURN tipo, cantidad
                ORDER BY cantidad DESC
            """).data()

            # Construir schema
            schema = SchemaOntologico()

            for tipo in tipos_entidad:
                schema.tipos_entidad[tipo['tipo']] = {
                    'cantidad': tipo['cantidad'],
                    'propiedades': tipo['propiedades_usadas'],
                    'activo': True
                }

            for tipo in tipos_relacion:
                schema.tipos_relacion[tipo['tipo']] = {
                    'cantidad': tipo['cantidad'],
                    'activo': True
                }

            # Estadísticas generales
            stats = session.run("""
                MATCH (e:Entidad)
                OPTIONAL MATCH (e)-[r]->()
                RETURN count(DISTINCT e) AS total_entidades,
                       count(r) AS total_relaciones
            """).single()

            schema.estadisticas = {
                'total_entidades': stats['total_entidades'],
                'total_relaciones': stats['total_relaciones'],
                'tipos_entidad': len(schema.tipos_entidad),
                'tipos_relacion': len(schema.tipos_relacion)
            }

            self.schema_cache = schema
            self.ultima_actualizacion = datetime.now()

            return schema

    def actualizar_schema(self, nuevas_entidades: List[Entidad], nuevas_relaciones: List[Relacion]):
        """Actualiza el schema con nuevos tipos descubiertos"""
        schema = self.schema_cache or self.cargar_schema_actual()

        # Detectar nuevos tipos de entidad
        tipos_nuevos = set()
        for entidad in nuevas_entidades:
            if entidad.tipo not in schema.tipos_entidad:
                tipos_nuevos.add(entidad.tipo)
                schema.tipos_entidad[entidad.tipo] = {
                    'cantidad': 0,
                    'propiedades': [],
                    'activo': True,
                    'fecha_creacion': datetime.now().isoformat()
                }

        # Detectar nuevos tipos de relación
        for relacion in nuevas_relaciones:
            if relacion.tipo not in schema.tipos_relacion:
                schema.tipos_relacion[relacion.tipo] = {
                    'cantidad': 0,
                    'activo': True,
                    'fecha_creacion': datetime.now().isoformat()
                }

        if tipos_nuevos:
            print(f"📋 Nuevos tipos de entidad descubiertos: {', '.join(tipos_nuevos)}")

        self.schema_cache = schema
        return schema

    def inferir_relaciones_posibles(self, entidades: List[Entidad]) -> List[Relacion]:
        """Usa LLM para inferir relaciones entre entidades basándose en contexto"""
        if len(entidades) < 2:
            return []

        # Preparar contexto (limitar longitud de contexto para evitar JSON largos)
        contexto_entidades = "\n".join([
            f"- {e.nombre} ({e.tipo})"
            for e in entidades[:10]  # Limitar para no saturar el prompt
        ])

        prompt = f"""Analiza estas entidades y sugiere relaciones lógicas.

ENTIDADES:
{contexto_entidades}

TIPOS DE RELACIONES: REQUIERE, COMPLEMENTA, PERTENECE_A

Responde SOLO con JSON válido en UNA SOLA LÍNEA (sin saltos de línea):
{{"relaciones": [{{"origen": "nombre_entidad_1", "destino": "nombre_entidad_2", "tipo": "REQUIERE", "confianza": 0.85}}]}}

JSON:"""

        def hacer_llamada():
            return self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                )
            )

        try:
            response = llamar_llm_con_retry(hacer_llamada)

            if not response:
                return []

            texto = response.text.replace('```json', '').replace('```', '').strip()

            # Reparación robusta del JSON
            texto = self._reparar_json_relaciones(texto)

            json_match = re.search(r'\{[\s\S]*\}', texto)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))

                    relaciones = []
                    nombres_entidades = {e.nombre for e in entidades}

                    for r in data.get('relaciones', []):
                        if r.get('origen') in nombres_entidades and r.get('destino') in nombres_entidades:
                            relaciones.append(Relacion(
                                origen=r['origen'],
                                destino=r['destino'],
                                tipo=r.get('tipo', 'RELACIONA'),
                                propiedades={
                                    'justificacion': r.get('justificacion', 'Inferida automáticamente'),
                                    'inferida': True
                                },
                                confianza=r.get('confianza', 0.7)
                            ))

                    if relaciones:
                        print(f"   Relaciones inferidas: {len(relaciones)}")
                    return relaciones

                except json.JSONDecodeError:
                    # Intentar extracción manual como fallback
                    return self._extraer_relaciones_fallback(texto, entidades)

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"⚠️ Rate limit alcanzado en inferencia de relaciones (se omitirá)")
            else:
                print(f"⚠️ Error infiriendo relaciones: {e}")

        return []

    def _reparar_json_relaciones(self, texto: str) -> str:
        """Repara JSON de relaciones con errores comunes"""
        # Eliminar comas antes de ] o }
        texto = re.sub(r',\s*(\]|\})', r'\1', texto)

        # Eliminar caracteres de control
        texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', texto)

        # Escapar saltos de línea dentro de strings
        resultado = []
        in_string = False
        escape = False

        for char in texto:
            if escape:
                resultado.append(char)
                escape = False
                continue
            if char == '\\':
                resultado.append(char)
                escape = True
                continue
            if char == '"':
                resultado.append(char)
                in_string = not in_string
                continue
            if in_string:
                if char == '\n':
                    resultado.append('\\n')
                elif char == '\r':
                    resultado.append('\\r')
                elif char == '\t':
                    resultado.append('\\t')
                else:
                    resultado.append(char)
            else:
                resultado.append(char)

        texto = ''.join(resultado)

        # Agregar comas faltantes entre objetos
        texto = re.sub(r'\}\s*\{', '}, {', texto)
        texto = re.sub(r'"\s*\n\s*"', '", "', texto)
        texto = re.sub(r'"\s*\n(\s*)\{', '",\n\1{', texto)

        return texto

    def _extraer_relaciones_fallback(self, texto: str, entidades: List[Entidad]) -> List[Relacion]:
        """Extrae relaciones manualmente si el JSON está muy roto"""
        relaciones = []
        nombres_entidades = {e.nombre for e in entidades}

        # Buscar patrones de relaciones: "origen": "...", "destino": "...", "tipo": "..."
        patron = r'"origen"\s*:\s*"([^"]+)"[^}]*"destino"\s*:\s*"([^"]+)"[^}]*"tipo"\s*:\s*"([^"]+)"'
        matches = re.finditer(patron, texto)

        for match in matches:
            origen = match.group(1)
            destino = match.group(2)
            tipo = match.group(3)

            if origen in nombres_entidades and destino in nombres_entidades:
                relaciones.append(Relacion(
                    origen=origen,
                    destino=destino,
                    tipo=tipo,
                    propiedades={'inferida': True, 'extraido_fallback': True},
                    confianza=0.6
                ))

        if relaciones:
            print(f"   Relaciones extraídas (fallback): {len(relaciones)}")

        return relaciones

    def generar_prompt_schema_aware(self, schema: SchemaOntologico) -> str:
        """Genera un prompt que incluye el schema actual para guiar la extracción"""
        tipos_str = ", ".join(list(schema.tipos_entidad.keys())[:15])
        relaciones_str = ", ".join(list(schema.tipos_relacion.keys())[:10])

        return f"""
SCHEMA ACTUAL DE LA BASE DE CONOCIMIENTO:

Tipos de Entidad Existentes ({len(schema.tipos_entidad)} tipos):
{tipos_str}

Tipos de Relación Existentes ({len(schema.tipos_relacion)} tipos):
{relaciones_str}

Estadísticas:
- Total entidades: {schema.estadisticas.get('total_entidades', 0)}
- Total relaciones: {schema.estadisticas.get('total_relaciones', 0)}

INSTRUCCIONES ONTOLÓGICAS:
1. PRIORIZA usar los tipos de entidad existentes
2. Si necesitas crear un tipo nuevo, debe ser significativamente diferente
3. Normaliza los nombres siguiendo el patrón: categoria_descripcion_corta
4. Las relaciones deben usar los tipos existentes cuando sea posible
"""


# ============================================================================
# RESOLUCIÓN DE ENTIDADES
# ============================================================================

class ResolucionEntidades:
    """Detecta y resuelve entidades duplicadas usando embeddings y LLM"""

    def __init__(self, driver, embedding_model, client: genai.Client):
        self.driver = driver
        self.embedding_model = embedding_model
        self.client = client

    def buscar_entidades_similares(
        self,
        nombre: str,
        contexto: str,
        tipo: str,
        umbral_similitud: float = 0.85
    ) -> List[Dict]:
        """Busca entidades que podrían ser duplicados"""

        # Generar embedding de la nueva entidad
        query_embedding = self.embedding_model.encode(contexto).tolist()

        with self.driver.session() as session:
            # Búsqueda híbrida: nombre similar + embedding similar + mismo tipo
            result = session.run("""
                // Búsqueda vectorial
                CALL db.index.vector.queryNodes(
                    'entidad_embedding',
                    10,
                    $query_embedding
                )
                YIELD node, score
                WHERE node.tipo = $tipo
                AND score > $umbral
                RETURN node.nombre AS nombre,
                       node.tipo AS tipo,
                       node.contexto AS contexto,
                       node.propiedades AS propiedades,
                       node.embedding AS embedding,
                       score AS similitud_vectorial
                ORDER BY score DESC
                LIMIT 5
            """,
                query_embedding=query_embedding,
                tipo=tipo,
                umbral=umbral_similitud
            )

            candidatos = []
            for record in result:
                propiedades = record["propiedades"]
                if isinstance(propiedades, str):
                    propiedades = json.loads(propiedades)

                # Calcular similitud de nombre (Levenshtein aproximado)
                similitud_nombre = self._similitud_cadenas(nombre, record["nombre"])

                candidatos.append({
                    "nombre": record["nombre"],
                    "tipo": record["tipo"],
                    "contexto": record["contexto"],
                    "propiedades": propiedades,
                    "similitud_vectorial": record["similitud_vectorial"],
                    "similitud_nombre": similitud_nombre,
                    "score_combinado": (record["similitud_vectorial"] * 0.7 + similitud_nombre * 0.3)
                })

            return sorted(candidatos, key=lambda x: x['score_combinado'], reverse=True)

    def _similitud_cadenas(self, s1: str, s2: str) -> float:
        """Calcula similitud entre dos cadenas (Jaro-Winkler simplificado)"""
        s1 = s1.lower().replace('_', ' ')
        s2 = s2.lower().replace('_', ' ')

        # Palabras en común
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        interseccion = words1.intersection(words2)
        union = words1.union(words2)

        return len(interseccion) / len(union) if union else 0.0

    def verificar_duplicado_con_llm(
        self,
        entidad_nueva: Entidad,
        candidatos: List[Dict]
    ) -> Tuple[bool, Optional[Dict]]:
        """Usa LLM para determinar si la entidad nueva es duplicado de algún candidato"""

        if not candidatos:
            return False, None

        # Preparar contexto para LLM
        candidatos_str = "\n".join([
            f"{i+1}. {c['nombre']} - Score: {c['score_combinado']:.2f}\n   Contexto: {c['contexto'][:150]}"
            for i, c in enumerate(candidatos[:3])
        ])

        prompt = f"""Determina si esta nueva entidad es DUPLICADO de alguna entidad existente.

NUEVA ENTIDAD:
- Nombre: {entidad_nueva.nombre}
- Tipo: {entidad_nueva.tipo}
- Contexto: {entidad_nueva.contexto}

CANDIDATOS SIMILARES:
{candidatos_str}

CRITERIOS:
- Son duplicados si se refieren al MISMO concepto/criterio/requisito
- NO son duplicados si son conceptos relacionados pero distintos
- Considera el contexto semántico, no solo el nombre

Responde SOLO con JSON (sin markdown):
{{
  "es_duplicado": true/false,
  "candidato_seleccionado": "nombre_del_candidato" o null,
  "justificacion": "breve explicación",
  "confianza": 0.0-1.0
}}
"""

        def hacer_llamada():
            return self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )

        try:
            response = llamar_llm_con_retry(hacer_llamada)

            if not response:
                return False, None

            texto = response.text.replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{[\s\S]*\}', texto)
            if json_match:
                resultado = json.loads(json_match.group(0))

                if resultado.get('es_duplicado', False) and resultado.get('confianza', 0) > 0.7:
                    # Encontrar el candidato seleccionado
                    nombre_seleccionado = resultado.get('candidato_seleccionado')
                    candidato = next((c for c in candidatos if c['nombre'] == nombre_seleccionado), None)
                    return True, candidato

        except Exception as e:
            print(f"⚠️ Error en verificación LLM: {e}")

        return False, None

    def fusionar_entidades(
        self,
        entidad_nueva: Entidad,
        entidad_existente: Dict
    ) -> Entidad:
        """Fusiona dos entidades duplicadas, enriqueciendo la existente"""

        print(f"🔗 Fusionando '{entidad_nueva.nombre}' → '{entidad_existente['nombre']}'")

        # La entidad nueva se convierte en alias de la existente
        entidad_fusionada = Entidad(
            nombre=entidad_existente['nombre'],  # Mantener el nombre canónico
            tipo=entidad_existente['tipo'],
            propiedades=entidad_existente['propiedades'].copy(),
            contexto=entidad_existente['contexto'],
            validada=True,
            entidad_canonica=entidad_existente['nombre']
        )

        # Enriquecer propiedades si la nueva entidad aporta información
        if entidad_nueva.propiedades:
            for key, value in entidad_nueva.propiedades.items():
                if key not in entidad_fusionada.propiedades and value:
                    entidad_fusionada.propiedades[key] = value

        # Registrar alias
        if 'aliases' not in entidad_fusionada.propiedades:
            entidad_fusionada.propiedades['aliases'] = []

        if entidad_nueva.nombre not in entidad_fusionada.propiedades['aliases']:
            entidad_fusionada.propiedades['aliases'].append(entidad_nueva.nombre)

        return entidad_fusionada


# ============================================================================
# VALIDADOR DE CONSISTENCIA
# ============================================================================

class ValidadorConsistencia:
    """Valida la consistencia ontológica del grafo y genera métricas"""

    def __init__(self, driver):
        self.driver = driver

    def calcular_metricas(self) -> Dict[str, Any]:
        """Calcula métricas completas de calidad ontológica"""

        with self.driver.session() as session:
            # 1. Entidades huérfanas (sin relaciones)
            huerfanas = session.run("""
                MATCH (e:Entidad)
                WHERE NOT (e)-[]-()
                RETURN count(e) AS cantidad,
                       collect(e.nombre)[..10] AS ejemplos
            """).single()

            # 2. Posibles duplicados (alta similitud vectorial)
            duplicados = session.run("""
                MATCH (e1:Entidad), (e2:Entidad)
                WHERE elementId(e1) < elementId(e2)
                AND e1.tipo = e2.tipo
                AND gds.similarity.cosine(e1.embedding, e2.embedding) > 0.95
                RETURN count(*) AS cantidad,
                       collect([e1.nombre, e2.nombre])[..5] AS ejemplos
            """).single()

            # 3. Distribución de tipos
            distribucion_tipos = session.run("""
                MATCH (e:Entidad)
                RETURN e.tipo AS tipo,
                       count(*) AS cantidad
                ORDER BY cantidad DESC
            """).data()

            # 4. Densidad del grafo
            densidad = session.run("""
                MATCH (e:Entidad)
                OPTIONAL MATCH (e)-[r]-()
                WITH count(DISTINCT e) AS nodos, count(r) AS relaciones
                RETURN nodos, relaciones,
                       CASE WHEN nodos > 1
                            THEN toFloat(relaciones) / (nodos * (nodos - 1))
                            ELSE 0.0
                       END AS densidad
            """).single()

            # 5. Nodos más conectados
            nodos_centrales = session.run("""
                MATCH (e:Entidad)-[r]-()
                WITH e, count(r) AS grado
                ORDER BY grado DESC
                LIMIT 10
                RETURN e.nombre AS nombre, grado
            """).data()

            # 6. Tipos de relación más usados
            relaciones_top = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS tipo, count(*) AS cantidad
                ORDER BY cantidad DESC
                LIMIT 10
            """).data()

            metricas = {
                "entidades_huerfanas": {
                    "cantidad": huerfanas["cantidad"],
                    "ejemplos": huerfanas["ejemplos"]
                },
                "posibles_duplicados": {
                    "cantidad": duplicados["cantidad"],
                    "ejemplos": duplicados["ejemplos"]
                },
                "distribucion_tipos": distribucion_tipos,
                "densidad_grafo": {
                    "nodos": densidad["nodos"],
                    "relaciones": densidad["relaciones"],
                    "densidad": round(densidad["densidad"], 4)
                },
                "nodos_centrales": nodos_centrales,
                "relaciones_top": relaciones_top,
                "timestamp": datetime.now().isoformat()
            }

            return metricas

    def generar_reporte_calidad(self, metricas: Dict) -> str:
        """Genera un reporte legible de la calidad ontológica"""

        reporte = f"""
{'='*80}
REPORTE DE CALIDAD ONTOLÓGICA
Fecha: {metricas['timestamp']}
{'='*80}

1. SALUD DEL GRAFO
   - Nodos totales: {metricas['densidad_grafo']['nodos']}
   - Relaciones totales: {metricas['densidad_grafo']['relaciones']}
   - Densidad: {metricas['densidad_grafo']['densidad']:.4f}

2. ENTIDADES HUÉRFANAS (sin conexiones)
   - Cantidad: {metricas['entidades_huerfanas']['cantidad']}
   - Ejemplos: {', '.join(metricas['entidades_huerfanas']['ejemplos'][:5])}

3. POSIBLES DUPLICADOS
   - Cantidad: {metricas['posibles_duplicados']['cantidad']}
   - Requieren revisión: {metricas['posibles_duplicados']['ejemplos'][:3]}

4. DISTRIBUCIÓN DE TIPOS
"""
        for tipo in metricas['distribucion_tipos'][:10]:
            reporte += f"   - {tipo['tipo']}: {tipo['cantidad']} entidades\n"

        reporte += f"""
5. NODOS MÁS CONECTADOS (Centrales)
"""
        for nodo in metricas['nodos_centrales'][:5]:
            reporte += f"   - {nodo['nombre']}: {nodo['grado']} conexiones\n"

        reporte += f"""
6. RELACIONES MÁS USADAS
"""
        for rel in metricas['relaciones_top'][:5]:
            reporte += f"   - {rel['tipo']}: {rel['cantidad']} veces\n"

        reporte += f"\n{'='*80}\n"

        return reporte

    def detectar_anomalias(self, metricas: Dict) -> List[str]:
        """Detecta anomalías en el grafo"""
        anomalias = []

        # Demasiadas entidades huérfanas
        porcentaje_huerfanas = (metricas['entidades_huerfanas']['cantidad'] /
                                metricas['densidad_grafo']['nodos'] * 100
                                if metricas['densidad_grafo']['nodos'] > 0 else 0)

        if porcentaje_huerfanas > 30:
            anomalias.append(f"⚠️ Alto porcentaje de entidades huérfanas: {porcentaje_huerfanas:.1f}%")

        # Muchos duplicados potenciales
        if metricas['posibles_duplicados']['cantidad'] > 10:
            anomalias.append(f"⚠️ {metricas['posibles_duplicados']['cantidad']} pares de posibles duplicados")

        # Baja densidad
        if metricas['densidad_grafo']['densidad'] < 0.01 and metricas['densidad_grafo']['nodos'] > 20:
            anomalias.append(f"⚠️ Densidad muy baja: {metricas['densidad_grafo']['densidad']:.4f}")

        # Tipos de entidad poco balanceados
        if metricas['distribucion_tipos']:
            tipo_dominante = metricas['distribucion_tipos'][0]
            total = sum(t['cantidad'] for t in metricas['distribucion_tipos'])
            porcentaje_dominante = (tipo_dominante['cantidad'] / total * 100) if total > 0 else 0

            if porcentaje_dominante > 70:
                anomalias.append(f"⚠️ Tipo '{tipo_dominante['tipo']}' domina el grafo ({porcentaje_dominante:.1f}%)")

        return anomalias


# ============================================================================
# AGENTE VALIDADOR ONTOLÓGICO
# ============================================================================

class AgenteValidadorOntologico:
    """Agente que valida y normaliza entidades antes de persistir"""

    def __init__(
        self,
        config: ConfiguracionColaba,
        ontologia_dinamica: OntologiaDinamica,
        resolucion_entidades: ResolucionEntidades
    ):
        self.config = config
        self.ontologia = ontologia_dinamica
        self.resolucion = resolucion_entidades
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)

        self.agent = Agent(
            name="validador_ontologico",
            model="gemini-2.5-flash",
            instruction="""
            Eres un validador ontológico experto. Aseguras que las entidades
            extraídas sean consistentes con el schema existente y detectas duplicados.
            """,
            description="Valida y normaliza entidades ontológicas"
        )

    def validar_entidad(self, entidad: Entidad) -> ResultadoValidacion:
        """Valida una entidad individual con validación inteligente para reducir llamadas LLM.
        
        ESTRATEGIA DE OPTIMIZACIÓN:
        - Score > 0.95: Asumir duplicado SIN llamar a LLM (muy alta similitud)
        - Score 0.85-0.95: Caso AMBIGUO, usar LLM para decidir
        - Score < 0.85: Asumir entidad nueva SIN llamar a LLM
        """

        # 1. Buscar entidades similares (usa embeddings, NO LLM)
        similares = self.resolucion.buscar_entidades_similares(
            nombre=entidad.nombre,
            contexto=entidad.contexto,
            tipo=entidad.tipo,
            umbral_similitud=0.80  # Bajamos umbral para capturar más candidatos
        )

        # 2. Validación inteligente basada en score de similitud
        if similares:
            mejor_candidato = similares[0]
            score = mejor_candidato['score_combinado']
            
            # CASO 1: Muy alta similitud (>0.95) - Asumir duplicado SIN LLM
            if score > 0.95:
                print(f"   ⚡ Auto-fusión (score={score:.2f}): '{entidad.nombre}' → '{mejor_candidato['nombre']}'")
                entidad_fusionada = self.resolucion.fusionar_entidades(entidad, mejor_candidato)
                return ResultadoValidacion(
                    entidad_validada=entidad_fusionada,
                    es_nueva=False,
                    entidades_fusionadas=[mejor_candidato['nombre']],
                    cambios_aplicados=[
                        f"Auto-fusionada (alta similitud {score:.2f})",
                        f"Nombre canónico: '{entidad_fusionada.nombre}'"
                    ],
                    score_confianza=score
                )
            
            # CASO 2: Similitud ambigua (0.85-0.95) - Usar LLM para decidir
            elif score >= 0.85:
                es_duplicado, candidato = self.resolucion.verificar_duplicado_con_llm(
                    entidad, similares
                )

                if es_duplicado and candidato:
                    entidad_fusionada = self.resolucion.fusionar_entidades(entidad, candidato)
                    return ResultadoValidacion(
                        entidad_validada=entidad_fusionada,
                        es_nueva=False,
                        entidades_fusionadas=[candidato['nombre']],
                        cambios_aplicados=[
                            f"Fusionada con '{candidato['nombre']}' (verificado por LLM)",
                            f"Nombre canónico: '{entidad_fusionada.nombre}'"
                        ],
                        score_confianza=candidato['score_combinado']
                    )
            
            # CASO 3: Baja similitud (<0.85) - Entidad nueva, NO llamar LLM
            # Se continúa al flujo normal de crear entidad nueva

        # 3. Normalizar nombre si es necesario
        entidad_normalizada = self._normalizar_nombre(entidad)

        # 4. Validar contra schema
        schema = self.ontologia.schema_cache or self.ontologia.cargar_schema_actual()
        es_tipo_nuevo = entidad_normalizada.tipo not in schema.tipos_entidad

        cambios = []
        if entidad_normalizada.nombre != entidad.nombre:
            cambios.append(f"Nombre normalizado: '{entidad.nombre}' → '{entidad_normalizada.nombre}'")

        if es_tipo_nuevo:
            cambios.append(f"Nuevo tipo de entidad: '{entidad_normalizada.tipo}'")

        return ResultadoValidacion(
            entidad_validada=entidad_normalizada,
            es_nueva=True,
            entidades_fusionadas=[],
            cambios_aplicados=cambios,
            score_confianza=1.0
        )

    def _normalizar_nombre(self, entidad: Entidad) -> Entidad:
        """Normaliza el nombre de la entidad según convenciones"""
        nombre_original = entidad.nombre

        # Convertir a snake_case
        nombre = nombre_original.lower()
        nombre = re.sub(r'[^\w\s]', '', nombre)  # Eliminar caracteres especiales
        nombre = re.sub(r'\s+', '_', nombre)      # Espacios a guiones bajos
        nombre = re.sub(r'_+', '_', nombre)       # Múltiples guiones a uno
        nombre = nombre.strip('_')                 # Eliminar guiones al inicio/fin

        # Limitar longitud
        if len(nombre) > 80:
            nombre = nombre[:80]

        entidad.nombre = nombre
        return entidad

    def validar_ontologia(self, ontologia: Ontologia) -> Tuple[Ontologia, List[ResultadoValidacion]]:
        """Valida una ontología completa"""
        print(f"\n{'='*80}")
        print(f"[Validador Ontológico] Validando ontología")
        print(f"{'='*80}\n")

        entidades_validadas = []
        resultados_validacion = []

        print(f"🔍 Validando {len(ontologia.entidades)} entidades...")

        for i, entidad in enumerate(ontologia.entidades, 1):
            if i % 10 == 0:
                print(f"   Procesadas: {i}/{len(ontologia.entidades)}")

            resultado = self.validar_entidad(entidad)
            resultados_validacion.append(resultado)

            # Solo añadir si no es duplicado
            if resultado.es_nueva or not resultado.entidades_fusionadas:
                entidades_validadas.append(resultado.entidad_validada)

        # Estadísticas de validación
        nuevas = sum(1 for r in resultados_validacion if r.es_nueva)
        fusionadas = sum(1 for r in resultados_validacion if r.entidades_fusionadas)

        print(f"\n✓ Validación completada:")
        print(f"  - Entidades nuevas: {nuevas}")
        print(f"  - Entidades fusionadas: {fusionadas}")
        print(f"  - Total a persistir: {len(entidades_validadas)}")

        ontologia_validada = Ontologia(
            entidades=entidades_validadas,
            relaciones=ontologia.relaciones,
            metadata={
                **ontologia.metadata,
                'validacion': {
                    'entidades_originales': len(ontologia.entidades),
                    'entidades_validadas': len(entidades_validadas),
                    'entidades_fusionadas': fusionadas,
                    'fecha_validacion': datetime.now().isoformat()
                }
            }
        )

        return ontologia_validada, resultados_validacion


# ============================================================================
# AGENTE 1: ONTÓLOGO (MEJORADO - SCHEMA-AWARE)
# ============================================================================

class AgenteOntologo:
    """Agente que extrae ontologías consciente del schema existente"""

    def __init__(self, config: ConfiguracionColaba, ontologia_dinamica: OntologiaDinamica):
        self.config = config
        self.ontologia = ontologia_dinamica
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)

        self.agent = Agent(
            name="ontologo",
            model="gemini-2.5-flash",
            instruction="""
            Eres un experto en análisis de documentos normativos académicos.
            Extraes ontologías estructuradas RESPETANDO el schema existente.
            Tu prioridad es generar JSON sintácticamente perfecto.
            """,
            description="Extrae ontologías con conciencia del schema"
        )

    def analizar_documento(self, documento: str, tipo_documento: str) -> Ontologia:
        """Analiza un documento respetando el schema existente"""
        print(f"\n{'='*80}")
        print(f"[Agente Ontólogo] Analizando documento tipo: {tipo_documento}")
        print(f"{'='*80}\n")

        # Cargar schema actual
        schema = self.ontologia.cargar_schema_actual()

        # Construir prompt schema-aware
        prompt = self._construir_prompt_schema_aware(documento, tipo_documento, schema)

        # Llamar a Gemini
        print("🔄 Enviando documento a Gemini 2.5 Flash (schema-aware)...")

        def hacer_llamada():
            return self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8000,
                )
            )

        try:
            response = llamar_llm_con_retry(hacer_llamada)

            if not response:
                raise Exception("No se obtuvo respuesta de Gemini")

            # Parsear respuesta
            ontologia_json = self._parsear_respuesta(response.text)

            # Si no se obtuvo nada válido, usar método de fallback
            if not ontologia_json.get("entidades"):
                print("⚠️ No se extrajeron entidades válidas, usando método de fallback...")
                ontologia_json = self._extraer_fallback(documento, tipo_documento, schema)

        except Exception as e:
            print(f"❌ Error en generación: {e}")
            print("🔄 Usando método de fallback...")
            ontologia_json = self._extraer_fallback(documento, tipo_documento, schema)

        # Convertir a objetos
        ontologia = self._convertir_a_ontologia(ontologia_json, tipo_documento)

        print(f"\n✓ Ontología generada:")
        print(f"  - {len(ontologia.entidades)} entidades")
        print(f"  - {len(ontologia.relaciones)} relaciones")

        return ontologia

    def _extraer_fallback(self, documento: str, tipo: str, schema: SchemaOntologico) -> Dict:
        """Método de fallback: extrae entidades de forma más simple"""
        print("🔧 Ejecutando extracción simplificada...")

        prompt_simple = f"""Analiza este material académico y extrae SOLO las entidades principales.

MATERIAL:
{documento[:2000]}

Extrae 5-10 conceptos clave como entidades. Usa tipos como: concepto, definicion, formula, proceso, ejemplo, tema.
Responde en formato JSON SIMPLE:

{{
  "entidades": [
    {{"nombre": "concepto_ejemplo", "tipo": "concepto", "propiedades": {{"descripcion": "breve", "materia": "nombre_materia"}}, "contexto": "texto breve"}},
    {{"nombre": "formula_ejemplo", "tipo": "formula", "propiedades": {{"descripcion": "breve", "materia": "nombre_materia"}}, "contexto": "texto breve"}}
  ],
  "relaciones": []
}}

SOLO JSON, sin markdown:"""

        def hacer_llamada():
            return self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_simple,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                )
            )

        try:
            response = llamar_llm_con_retry(hacer_llamada)

            if not response:
                raise Exception("No se obtuvo respuesta en fallback")

            resultado = self._parsear_respuesta(response.text)

            if resultado.get("entidades"):
                print(f"✓ Fallback exitoso: {len(resultado['entidades'])} entidades")
                return resultado

        except Exception as e:
            print(f"⚠️ Fallback también falló: {e}")

        # Último recurso: crear manualmente
        print("⚠️ Creando ontología mínima manual...")
        return {
            "entidades": [
                {
                    "nombre": f"entidad_{tipo}_general",
                    "tipo": "criterio_evaluacion",
                    "propiedades": {
                        "descripcion": "Entidad extraída por método de fallback",
                        "obligatorio": False
                    },
                    "contexto": documento[:150]
                }
            ],
            "relaciones": []
        }

    def _construir_prompt_schema_aware(
        self,
        documento: str,
        tipo: str,
        schema: SchemaOntologico
    ) -> str:
        """Construye el prompt incluyendo el schema existente"""

        schema_info = self.ontologia.generar_prompt_schema_aware(schema)

        return f"""Analiza este material académico y extrae su ontología RESPETANDO EL SCHEMA EXISTENTE.

{schema_info}

TIPO DE MATERIAL: {tipo}

DOCUMENTO/MATERIAL:
{documento[:4000]}

TIPOS DE ENTIDAD SUGERIDOS (usa estos o los del schema existente):
- concepto: Ideas, teorías, principios fundamentales de la materia
- definicion: Definiciones formales de términos técnicos
- formula: Ecuaciones, fórmulas matemáticas o científicas
- proceso: Procedimientos, algoritmos, metodologías paso a paso
- ejemplo: Casos de estudio, ejemplos prácticos, ejercicios
- requisito: Prerrequisitos, conocimientos previos necesarios
- objetivo_aprendizaje: Competencias, habilidades a desarrollar
- criterio_evaluacion: Criterios para evaluar el aprendizaje
- recurso: Bibliografía, materiales, herramientas recomendadas
- tema: Unidades temáticas, módulos, secciones del contenido

INSTRUCCIONES DE EXTRACCIÓN:
1. PRIORIZA usar los tipos de entidad que YA EXISTEN en el schema
2. Solo crea un nuevo tipo si es conceptualmente diferente a los existentes
3. Normaliza nombres siguiendo el patrón: categoria_concepto (ej: concepto_cinematica_lineal)
4. Para cada entidad extrae:
   - nombre: identificador único normalizado (sin espacios, sin caracteres especiales)
   - tipo: usar tipos existentes preferentemente
   - propiedades: objeto JSON con descripcion, materia, nivel_dificultad (basico/intermedio/avanzado)
   - contexto: texto original (máximo 200 caracteres, sin saltos de línea reales)
5. Identifica RELACIONES (CRÍTICO: MÍNIMO 3 RELACIONES POR ENTIDAD):
   - Identifica TODAS las relaciones explícitas e IMPLÍCITAS entre las entidades.
   - Conecta los conceptos densamente. No dejes entidades aisladas.
   - Usa verbos precisos para 'tipo': REQUIERE, COMPLEMENTA, DEFINE, EJEMPLIFICA, PERTENECE_A, ES_PARTE_DE, ANTECEDE_A.
   - Si un concepto A es fundamental para entender B, crea la relación: B --REQUIERE--> A.
   - Si un ejemplo ilustra un tema, crea: Ejemplo --EJEMPLIFICA--> Tema.

FORMATO DE SALIDA - CRÍTICO PARA EVITAR ERRORES:
- Responde SOLO con JSON válido.
- NO uses markdown (```json).
- NO incluyas comentarios // ni #.
- TODAS las claves y strings deben usar comillas dobles (").
- ASEGURA EXPLICITAMENTE poner comas (,) entre cada elemento de lista y cada par clave-valor.
- NO debe haber comas al final de listas o objetos (trailing commas).
- ESCAPA correctamente las comillas dobles dentro de strings (\\"ejemplo\\").
- ESCAPA los saltos de línea dentro de strings (\\n).
- El contexto debe ser plano, sin saltos de línea reales.

ESTRUCTURA EXACTA (copia este formato y VERIFICA las COMAS):
{{
  "entidades": [
    {{
      "nombre": "concepto_cinematica_lineal",
      "tipo": "concepto",
      "propiedades": {{
        "descripcion": "Estudio del movimiento en linea recta",
        "materia": "Fisica",
        "nivel_dificultad": "basico"
      }},
      "contexto": "La cinematica lineal estudia el movimiento de objetos..."
    }}
  ],
  "relaciones": [
    {{
      "origen": "formula_velocidad",
      "destino": "concepto_cinematica_lineal",
      "tipo": "PERTENECE_A",
      "propiedades": {{}}
    }}
  ]
}}

RESPONDE AHORA SOLO CON EL JSON LIMPIO:"""

    def _parsear_respuesta(self, respuesta: str) -> Dict:
        """Parsea la respuesta JSON de Gemini con manejo robusto de errores"""
        # Limpiar markdown y espacios
        respuesta = respuesta.replace('```json', '').replace('```', '').strip()

        # Extraer JSON
        json_match = re.search(r'\{[\s\S]*\}', respuesta)

        if not json_match:
            print("⚠️ No se encontró JSON en la respuesta")
            return {"entidades": [], "relaciones": []}

        json_str = json_match.group(0)

        # Intentar parsear directamente
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parseando JSON (intento 1): {e}")

        # Aplicar reparaciones comunes
        try:
            json_str_fixed = self._reparar_json_comun(json_str)
            return json.loads(json_str_fixed)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parseando JSON (intento 2): {e}")

        # Reparación avanzada con regex
        try:
            json_str_advanced = self._reparar_json_avanzado(json_str)
            result = json.loads(json_str_advanced)
            print("✓ JSON reparado exitosamente con regex")
            return result
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parseando JSON (intento 3): {e}")

        # Intento 4: Extracción parcial de entidades
        try:
            result = self._extraer_json_parcial(json_str)
            if result.get("entidades"):
                print(f"✓ JSON parcial extraído: {len(result['entidades'])} entidades")
                return result
        except Exception as e:
            print(f"⚠️ Error en extracción parcial: {e}")

        # Último recurso: usar Gemini para reparar (con retry)
        try:
            return self._reparar_json_con_llm(json_str)
        except Exception as e:
            print(f"⚠️ Fallo total en reparación de JSON: {e}")
            return {"entidades": [], "relaciones": []}

    def _reparar_json_comun(self, json_str: str) -> str:
        """Aplica reparaciones comunes de JSON"""
        # Eliminar comas antes de ] o }
        json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)

        # Eliminar caracteres de control (excepto \n, \r, \t)
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)

        # Eliminar comas múltiples
        json_str = re.sub(r',+', ',', json_str)

        # Agregar comas faltantes entre objetos de lista: } { -> }, {
        json_str = re.sub(r'\}\s*\{', '}, {', json_str)
        json_str = re.sub(r'\}\s*\n\s*\{', '},\n{', json_str)

        # Agregar comas faltantes entre propiedades: "val" "key" -> "val", "key"
        # Cuidado: solo si hay salto de línea o espacios claros
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)

        # Eliminar coma al final antes de cerrar
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json_str

    def _escapar_saltos_en_strings(self, json_str: str) -> str:
        """Escapa saltos de línea dentro de valores de string JSON"""
        resultado = []
        in_string = False
        escape = False
        i = 0

        while i < len(json_str):
            char = json_str[i]

            if escape:
                resultado.append(char)
                escape = False
                i += 1
                continue

            if char == '\\':
                resultado.append(char)
                escape = True
                i += 1
                continue

            if char == '"':
                resultado.append(char)
                in_string = not in_string
                i += 1
                continue

            if in_string:
                # Dentro de un string, escapar caracteres problemáticos
                if char == '\n':
                    resultado.append('\\n')
                elif char == '\r':
                    resultado.append('\\r')
                elif char == '\t':
                    resultado.append('\\t')
                else:
                    resultado.append(char)
            else:
                resultado.append(char)

            i += 1

        return ''.join(resultado)

    def _agregar_comas_faltantes(self, json_str: str) -> str:
        """Agrega comas faltantes entre elementos de arrays y objetos"""
        # Patrón: "..." seguido de espacios/newlines y luego "..."
        # o }{ o ]" o }" etc sin coma entre ellos

        # Agregar coma entre strings consecutivos: "foo"\s+"bar" -> "foo",\s+"bar"
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)

        # También manejar sin newline
        json_str = re.sub(r'"\s{2,}"', '", "', json_str)

        # Agregar coma entre } y { consecutivos
        json_str = re.sub(r'\}\s*\n\s*\{', '},\n{', json_str)
        json_str = re.sub(r'\}\s+\{', '}, {', json_str)

        # Agregar coma entre ] y [ consecutivos
        json_str = re.sub(r'\]\s*\n\s*\[', '],\n[', json_str)

        # Agregar coma entre } y "
        json_str = re.sub(r'\}\s*\n\s*"', '},\n"', json_str)
        json_str = re.sub(r'\}\s+"', '}, "', json_str)

        # Agregar coma entre ] y "
        json_str = re.sub(r'\]\s*\n\s*"', '],\n"', json_str)

        # Agregar coma entre valor booleano/número y siguiente elemento
        json_str = re.sub(r'(true|false|null|\d+)\s*\n\s*"', r'\1,\n"', json_str)
        json_str = re.sub(r'(true|false|null|\d+)\s*\n\s*\{', r'\1,\n{', json_str)

        # Patrón específico para el error line 362: contexto largo seguido de nueva entidad
        # Buscar patrones como: "texto..."\n    {\n
        json_str = re.sub(r'"\s*\n(\s*)\{', r'",\n\1{', json_str)

        return json_str

    def _reparar_json_avanzado(self, json_str: str) -> str:
        """Reparación avanzada de JSON con manejo de strings rotos"""
        # 1. Primero escapar saltos de línea dentro de strings
        json_str = self._escapar_saltos_en_strings(json_str)

        # 2. Aplicar reparaciones comunes
        json_str = self._reparar_json_comun(json_str)

        # 3. Agregar comas faltantes
        json_str = self._agregar_comas_faltantes(json_str)

        # 4. Manejar comillas no escapadas dentro de valores de string
        lines = json_str.split('\n')
        fixed_lines = []

        for line in lines:
            # Si la línea contiene una clave-valor de string
            if ':' in line and '"' in line:
                # Dividir por el primer : para separar clave de valor
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0]
                    value_part = parts[1]

                    # Contar comillas en el valor (excluyendo las escapadas)
                    unescaped_quotes = len(re.findall(r'(?<!\\)"', value_part))

                    # Si hay más de 2 comillas (apertura y cierre), hay comillas no escapadas
                    if unescaped_quotes > 2:
                        # Encontrar el primer " y el último ", y reemplazar los del medio
                        first_quote = value_part.find('"')
                        last_quote = value_part.rfind('"')

                        if first_quote != -1 and last_quote != -1 and first_quote < last_quote:
                            # Reemplazar comillas entre la primera y la última
                            middle = value_part[first_quote+1:last_quote]
                            middle_fixed = re.sub(r'(?<!\\)"', "'", middle)
                            value_part = value_part[:first_quote+1] + middle_fixed + value_part[last_quote:]

                    line = key_part + ':' + value_part

            fixed_lines.append(line)

        json_str = '\n'.join(fixed_lines)

        return json_str

    def _extraer_json_parcial(self, json_str: str) -> Dict:
        """Extrae entidades individuales del JSON aunque esté roto"""
        entidades = []

        # Buscar patrones de entidades individuales
        # Patrón: {"nombre": "...", "tipo": "...", ...}
        patron_entidad = r'\{\s*"nombre"\s*:\s*"([^"]+)"\s*,\s*"tipo"\s*:\s*"([^"]+)"[^}]*"contexto"\s*:\s*"([^"]*)"[^}]*\}'
        matches = re.finditer(patron_entidad, json_str, re.DOTALL)

        for match in matches:
            try:
                nombre = match.group(1)
                tipo = match.group(2)
                contexto = match.group(3)[:200]  # Limitar contexto

                entidades.append({
                    "nombre": nombre,
                    "tipo": tipo,
                    "propiedades": {"extraido_parcialmente": True},
                    "contexto": contexto
                })
            except:
                continue

        # Si no encontramos con el patrón completo, intentar más simple
        if not entidades:
            patron_simple = r'"nombre"\s*:\s*"([^"]+)"'
            nombres = re.findall(patron_simple, json_str)

            patron_tipos = r'"tipo"\s*:\s*"([^"]+)"'
            tipos = re.findall(patron_tipos, json_str)

            for i, nombre in enumerate(nombres[:10]):  # Máximo 10
                tipo = tipos[i] if i < len(tipos) else "criterio_evaluacion"
                entidades.append({
                    "nombre": nombre,
                    "tipo": tipo,
                    "propiedades": {"extraido_parcialmente": True},
                    "contexto": f"Extraído automáticamente: {nombre}"
                })

        return {"entidades": entidades, "relaciones": []}

    def _reparar_json_con_llm(self, json_str: str) -> Dict:
        """Usa Gemini para reparar JSON malformado con retry"""
        print("🔧 Intentando reparar JSON con Gemini...")

        # Limitar tamaño para no saturar el prompt
        json_truncado = json_str[:4000]

        repair_prompt = f"""Repara este JSON malformado. El JSON debe tener esta estructura:

{{
  "entidades": [
    {{"nombre": "...", "tipo": "...", "propiedades": {{}}, "contexto": "..."}}
  ],
  "relaciones": [
    {{"origen": "...", "destino": "...", "tipo": "...", "propiedades": {{}}}}
  ]
}}

JSON MALFORMADO:
{json_truncado}

REGLAS:
1. Responde SOLO con el JSON corregido
2. NO incluyas markdown (```json) ni explicaciones
3. Asegúrate de que TODOS los strings estén entre comillas
4. NO dejes comas antes de ] o }}
5. Escapa caracteres especiales en strings (\\n, \\t, etc.)

JSON CORREGIDO:"""

        def hacer_llamada():
            return self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=repair_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8000,
                )
            )

        try:
            response = llamar_llm_con_retry(hacer_llamada)

            if not response:
                print("⚠️ Gemini no devolvió respuesta para reparación")
                return self._extraer_json_parcial(json_str)

            repaired = response.text.replace('```json', '').replace('```', '').strip()
            repaired_match = re.search(r'\{[\s\S]*\}', repaired)

            if repaired_match:
                result = json.loads(repaired_match.group(0))
                print("✓ JSON reparado exitosamente con Gemini")
                return result
            else:
                print("⚠️ Gemini no devolvió JSON válido")
                return self._extraer_json_parcial(json_str)

        except Exception as e:
            print(f"⚠️ Error en reparación con Gemini: {e}")
            return self._extraer_json_parcial(json_str)

        return {"entidades": [], "relaciones": []}

    def _convertir_a_ontologia(self, json_data: Dict, tipo_doc: str) -> Ontologia:
        """Convierte JSON a objetos Ontologia"""
        entidades = [
            Entidad(
                nombre=e["nombre"],
                tipo=e["tipo"],
                propiedades=e["propiedades"],
                contexto=e["contexto"]
            )
            for e in json_data.get("entidades", [])
        ]

        relaciones = [
            Relacion(
                origen=r["origen"],
                destino=r["destino"],
                tipo=r["tipo"],
                propiedades=r.get("propiedades", {})
            )
            for r in json_data.get("relaciones", [])
        ]

        return Ontologia(
            entidades=entidades,
            relaciones=relaciones,
            metadata={
                "tipo_documento": tipo_doc,
                "fecha_procesamiento": datetime.now().isoformat(),
                "num_entidades": len(entidades),
                "num_relaciones": len(relaciones)
            }
        )


# ============================================================================
# AGENTE 2: PERSISTENCIA (MEJORADO)
# ============================================================================

class AgentePersistencia:
    """Agente que gestiona la persistencia en Neo4j con ontología dinámica"""

    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )

        print("🔄 Cargando modelo de embeddings...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✓ Modelo de embeddings cargado")

        self._inicializar_bd()

        self.agent = Agent(
            name="persistencia",
            model="gemini-2.5-flash",
            instruction="Gestiona el almacenamiento de ontologías en Neo4j",
            description="Agente de persistencia con soporte ontológico"
        )

    def _inicializar_bd(self):
        """Crea índices y constraints en Neo4j"""
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT entidad_nombre IF NOT EXISTS
                FOR (e:Entidad) REQUIRE e.nombre IS UNIQUE
            """)

            try:
                session.run("""
                    CREATE FULLTEXT INDEX entidad_fulltext IF NOT EXISTS
                    FOR (e:Entidad) ON EACH [e.contexto, e.descripcion]
                """)
            except:
                pass

            try:
                session.run("""
                    CREATE VECTOR INDEX entidad_embedding IF NOT EXISTS
                    FOR (e:Entidad) ON (e.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine',
                            `vector.quantization.enabled`: true
                        }
                    }
                """)
                print("✓ Índice vectorial creado")
            except Exception as e:
                print(f"⚠️ Índice vectorial ya existe: {e}")

            print("✓ Base de datos Neo4j inicializada")

    def guardar_ontologia(self, ontologia: Ontologia) -> Dict[str, Any]:
        """Guarda la ontología en Neo4j"""
        print(f"\n{'='*80}")
        print(f"[Agente Persistencia] Guardando ontología en Neo4j")
        print(f"{'='*80}\n")

        with self.driver.session() as session:
            print("📝 Guardando entidades con embeddings...")
            entidades_creadas = 0
            for entidad in ontologia.entidades:
                self._guardar_entidad(session, entidad)
                entidades_creadas += 1

            print("🔗 Guardando relaciones...")
            relaciones_creadas = 0
            for relacion in ontologia.relaciones:
                self._guardar_relacion(session, relacion)
                relaciones_creadas += 1

            resultado = {
                "entidades_creadas": entidades_creadas,
                "relaciones_creadas": relaciones_creadas,
                "timestamp": datetime.now().isoformat(),
                "metadata": ontologia.metadata
            }

            print(f"\n✓ Persistencia completada:")
            print(f"  - {entidades_creadas} entidades guardadas")
            print(f"  - {relaciones_creadas} relaciones guardadas")

            return resultado

    def _guardar_entidad(self, session, entidad: Entidad):
        """Guarda una entidad con su embedding"""
        embedding = self.embedding_model.encode(entidad.contexto).tolist()

        session.run("""
            MERGE (e:Entidad {nombre: $nombre})
            SET e.tipo = $tipo,
                e.contexto = $contexto,
                e.embedding = $embedding,
                e.propiedades = $propiedades,
                e.validada = $validada,
                e.fecha_creacion = datetime(),
                e.fecha_actualizacion = datetime()
        """,
            nombre=entidad.nombre,
            tipo=entidad.tipo,
            contexto=entidad.contexto,
            embedding=embedding,
            propiedades=json.dumps(entidad.propiedades, ensure_ascii=False),
            validada=entidad.validada
        )

    def _guardar_relacion(self, session, relacion: Relacion):
        """Guarda una relación entre entidades"""
        query = f"""
            MATCH (origen:Entidad {{nombre: $origen}})
            MATCH (destino:Entidad {{nombre: $destino}})
            MERGE (origen)-[r:{relacion.tipo.upper()}]->(destino)
            SET r.propiedades = $propiedades,
                r.confianza = $confianza,
                r.fecha_creacion = datetime()
        """

        session.run(query,
            origen=relacion.origen,
            destino=relacion.destino,
            propiedades=json.dumps(relacion.propiedades, ensure_ascii=False),
            confianza=relacion.confianza
        )

    def busqueda_hibrida(self, query: str, k: int = 10) -> List[Dict]:
        """Búsqueda híbrida: vectorial + texto completo"""
        print(f"\n🔍 Búsqueda híbrida para: '{query}'")

        query_embedding = self.embedding_model.encode(query).tolist()
        resultados_vectorial = self._busqueda_vectorial(query_embedding, k)
        resultados_texto = self._busqueda_texto(query, k)
        resultados_combinados = self._combinar_resultados(
            resultados_vectorial,
            resultados_texto,
            k
        )

        print(f"✓ Encontrados {len(resultados_combinados)} resultados relevantes")
        return resultados_combinados

    def _busqueda_vectorial(self, query_embedding: List[float], k: int) -> List[Dict]:
        """Búsqueda por similitud vectorial"""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'entidad_embedding',
                    $k,
                    $query_embedding
                )
                YIELD node, score
                RETURN node.nombre AS nombre,
                       node.tipo AS tipo,
                       node.contexto AS contexto,
                       node.propiedades AS propiedades,
                       score AS score_vectorial
            """, k=k, query_embedding=query_embedding)

            entidades = []
            for record in result:
                propiedades = record["propiedades"]
                if isinstance(propiedades, str):
                    propiedades = json.loads(propiedades)

                entidades.append({
                    "nombre": record["nombre"],
                    "tipo": record["tipo"],
                    "contexto": record["contexto"],
                    "propiedades": propiedades,
                    "score_vectorial": record["score_vectorial"]
                })

            return entidades

    def _busqueda_texto(self, query: str, k: int) -> List[Dict]:
        """Búsqueda de texto completo"""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes('entidad_fulltext', $search_term)
                YIELD node, score
                RETURN node.nombre AS nombre,
                       node.tipo AS tipo,
                       node.contexto AS contexto,
                       node.propiedades AS propiedades,
                       score
                LIMIT $k
            """, search_term=query, k=k)

            return [{
                "nombre": record["nombre"],
                "tipo": record["tipo"],
                "contexto": record["contexto"],
                "propiedades": json.loads(record["propiedades"]),
                "score_texto": record["score"]
            } for record in result]

    def _combinar_resultados(self, res_vec: List, res_txt: List, k: int) -> List[Dict]:
        """Combina resultados vectoriales y de texto"""
        max_vec = max([r.get("score_vectorial", 0) for r in res_vec] or [1])
        max_txt = max([r.get("score_texto", 0) for r in res_txt] or [1])

        resultados = {}

        for r in res_vec:
            nombre = r["nombre"]
            resultados[nombre] = r.copy()
            resultados[nombre]["score_vectorial_norm"] = r["score_vectorial"] / max_vec
            resultados[nombre]["score_texto_norm"] = 0

        for r in res_txt:
            nombre = r["nombre"]
            if nombre not in resultados:
                resultados[nombre] = r.copy()
                resultados[nombre]["score_vectorial_norm"] = 0
            resultados[nombre]["score_texto_norm"] = r["score_texto"] / max_txt

        for nombre in resultados:
            resultados[nombre]["score_total"] = (
                0.6 * resultados[nombre]["score_vectorial_norm"] +
                0.4 * resultados[nombre]["score_texto_norm"]
            )

            # Obtener relaciones asociadas a cada entidad
            resultados[nombre]["relaciones"] = self._obtener_relaciones_entidad(nombre)

        resultados_lista = sorted(
            resultados.values(),
            key=lambda x: x["score_total"],
            reverse=True
        )

        # Si no hay resultados, intentar obtener todas las entidades
        if not resultados_lista:
            print("⚠️ No se encontraron resultados, obteniendo todas las entidades...")
            resultados_lista = self._obtener_todas_entidades(k)

        return resultados_lista[:k]

    def _obtener_relaciones_entidad(self, nombre_entidad: str) -> List[Dict]:
        """Obtiene todas las relaciones de una entidad"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entidad {nombre: $nombre})-[r]->(destino:Entidad)
                    RETURN type(r) AS tipo_relacion,
                           destino.nombre AS destino_nombre,
                           destino.tipo AS destino_tipo,
                           r.propiedades AS propiedades
                    UNION
                    MATCH (origen:Entidad)-[r]->(e:Entidad {nombre: $nombre})
                    RETURN type(r) AS tipo_relacion,
                           origen.nombre AS destino_nombre,
                           origen.tipo AS destino_tipo,
                           r.propiedades AS propiedades
                """, nombre=nombre_entidad)

                relaciones = []
                for record in result:
                    relaciones.append({
                        "tipo": record["tipo_relacion"],
                        "entidad_relacionada": record["destino_nombre"],
                        "tipo_entidad": record["destino_tipo"]
                    })
                return relaciones
        except Exception:
            return []

    def _obtener_todas_entidades(self, k: int) -> List[Dict]:
        """Obtiene todas las entidades cuando la búsqueda no encuentra resultados"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entidad)
                    RETURN e.nombre AS nombre,
                           e.tipo AS tipo,
                           e.contexto AS contexto,
                           e.propiedades AS propiedades
                    LIMIT $k
                """, k=k)

                entidades = []
                for record in result:
                    propiedades = record["propiedades"]
                    if isinstance(propiedades, str):
                        try:
                            propiedades = json.loads(propiedades)
                        except:
                            propiedades = {}

                    entidad = {
                        "nombre": record["nombre"],
                        "tipo": record["tipo"],
                        "contexto": record["contexto"] or "",
                        "propiedades": propiedades or {},
                        "score_total": 1.0,
                        "score_vectorial_norm": 1.0,
                        "score_texto_norm": 1.0,
                        "relaciones": self._obtener_relaciones_entidad(record["nombre"])
                    }
                    entidades.append(entidad)

                if entidades:
                    print(f"✓ Recuperadas {len(entidades)} entidades directamente")
                return entidades
        except Exception as e:
            print(f"⚠️ Error obteniendo entidades: {e}")
            return []

    def close(self):
        """Cierra la conexión a Neo4j"""
        self.driver.close()


# ============================================================================
# AGENTE 3: BÚSQUEDA (SIN CAMBIOS MAYORES)
# ============================================================================

class AgenteBusqueda:
    """Agente que procesa prompts de usuario y coordina búsquedas"""

    def __init__(self, config: ConfiguracionColaba, agente_persistencia: AgentePersistencia):
        self.config = config
        self.persistencia = agente_persistencia
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)

        self.agent = Agent(
            name="busqueda",
            model="gemini-2.5-flash",
            instruction="""
            Eres un agente especializado en procesar solicitudes de usuarios
            para generar rúbricas académicas. Analizas la intención del usuario
            y coordinas búsquedas en la base de conocimiento.
            """,
            description="Procesa prompts y realiza búsquedas híbridas"
        )

    def procesar_prompt(self, prompt_usuario: str) -> Dict[str, Any]:
        """Procesa el prompt y realiza búsqueda híbrida"""
        print(f"\n{'='*80}")
        print(f"[Agente Búsqueda] Procesando prompt de usuario")
        print(f"{'='*80}\n")
        print(f"📝 Prompt: '{prompt_usuario}'\n")

        analisis = self._analizar_intencion(prompt_usuario)
        query_expandida = self._expandir_query(prompt_usuario, analisis)
        resultados = self.persistencia.busqueda_hibrida(query_expandida, k=15)

        return {
            "prompt_original": prompt_usuario,
            "analisis_intencion": analisis,
            "query_expandida": query_expandida,
            "resultados": resultados,
            "num_resultados": len(resultados)
        }

    def _analizar_intencion(self, prompt: str) -> Dict:
        """Analiza la intención del usuario con Gemini"""
        print("🤔 Analizando intención del usuario...")

        prompt_analisis = f"""Analiza este prompt de un usuario que quiere generar una rúbrica académica:

PROMPT: "{prompt}"

Responde SOLO con JSON (sin markdown):
{{"tipo_documento": "guia_docente", "ambito_normativo": ["igualdad_genero"], "nivel_detalle": "detallado", "palabras_clave": ["lenguaje", "inclusivo"]}}

JSON:"""

        def hacer_llamada():
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_analisis,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                )
            )

            texto = response.text.replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{[\s\S]*\}', texto)
            if json_match:
                return json.loads(json_match.group(0))
            return {}

        try:
            resultado = llamar_llm_con_retry(hacer_llamada, max_intentos=3, backoff_base=5)
            return resultado if resultado else {}
        except Exception as e:
            print(f"⚠️ Error en análisis de intención: {e}")
            return {}

    def _expandir_query(self, query: str, analisis: Dict) -> str:
        """Expande la query con términos del análisis"""
        terminos = [query]

        if "palabras_clave" in analisis:
            terminos.extend(analisis["palabras_clave"][:5])

        query_expandida = " ".join(terminos)
        print(f"🔍 Query expandida: '{query_expandida}'")

        return query_expandida


# ============================================================================
# AGENTE 4: RUBRICADOR (SIN CAMBIOS)
# ============================================================================

class AgenteRubricador:
    """Agente que genera rúbricas usando RAG híbrido con Gemini"""

    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)

        self.agent = Agent(
            name="rubricador",
            model="gemini-2.5-flash",
            instruction="""
            Eres un experto en DISEÑO DE RÚBRICAS ACADÉMICAS para cualquier materia o disciplina.
            Tu objetivo es generar instrumentos de evaluación EXTENSOS, EXHAUSTIVOS y PROFESIONALES
            adaptados al contenido específico de la materia proporcionada.
            No generas simples listas, sino documentos pedagógicos completos y detallados.
            """,
            description="Genera rúbricas profesionales para cualquier materia académica"
        )

    def generar_rubrica(self, prompt_usuario: str, contexto_busqueda: Dict) -> str:
        """Genera una rúbrica completa usando RAG"""
        print(f"\n{'='*80}")
        print(f"[Agente Rubricador] Generando rúbrica con RAG híbrido")
        print(f"{'='*80}\n")

        contexto_rag = self._preparar_contexto_rag(contexto_busqueda["resultados"])

        print("✍️  Generando rúbrica...")
        rubrica = self._generar_con_gemini(prompt_usuario, contexto_rag, contexto_busqueda)

        print("\n✓ Rúbrica generada exitosamente")

        return rubrica

    def _preparar_contexto_rag(self, resultados: List[Dict]) -> str:
        """Prepara el contexto para RAG incluyendo relaciones"""
        if not resultados:
            return "No se encontraron criterios en la base de conocimiento. Generar rúbrica basada en mejores prácticas.\n"

        contexto = "CONOCIMIENTOS Y CONCEPTOS EXTRAÍDOS DE LA BASE DE CONOCIMIENTO:\n\n"

        for i, resultado in enumerate(resultados[:15], 1):
            contexto += f"{i}. [{resultado.get('tipo', 'criterio')}] {resultado.get('nombre', 'Sin nombre')}\n"

            score_total = resultado.get('score_total', 0)
            if score_total:
                contexto += f"   Relevancia: {score_total:.3f}\n"

            propiedades = resultado.get('propiedades', {})
            if isinstance(propiedades, dict) and 'descripcion' in propiedades:
                contexto += f"   Descripción: {propiedades['descripcion']}\n"

            contexto_texto = resultado.get('contexto', '')
            if contexto_texto:
                contexto += f"   Contexto: {contexto_texto[:300]}...\n"

            # Agregar relaciones si existen
            relaciones = resultado.get('relaciones', [])
            if relaciones:
                contexto += f"   Relaciones:\n"
                for rel in relaciones[:5]:  # Máximo 5 relaciones por entidad
                    contexto += f"     - {rel.get('tipo', 'RELACIONA')} → {rel.get('entidad_relacionada', 'N/A')} ({rel.get('tipo_entidad', 'N/A')})\n"

            contexto += "\n"

        # Agregar resumen de relaciones encontradas
        total_relaciones = sum(len(r.get('relaciones', [])) for r in resultados)
        contexto += f"\n--- RESUMEN: {len(resultados)} entidades, {total_relaciones} relaciones encontradas ---\n"

        return contexto

    def _generar_con_gemini(self, prompt: str, contexto_rag: str, info_busqueda: Dict) -> str:
        """Genera la rúbrica usando Gemini con RAG"""

        prompt_generacion = f"""Eres un ARQUITECTO PEDAGÓGICO experto en diseño de instrumentos de evaluación académica.

SOLICITUD DEL USUARIO:
{prompt}

CONTENIDO ACADÉMICO EXTRAÍDO DE LA BASE DE CONOCIMIENTO:
{contexto_rag}

TU TAREA:
Generar una RÚBRICA DE EVALUACIÓN EXTENSA y DETALLADA adaptada a la materia y contenido proporcionado.
El documento debe ser exhaustivo, abarcando todos los aspectos pedagógicos relevantes.

ESTRUCTURA OBLIGATORIA DEL DOCUMENTO:

1. INFORMACIÓN GENERAL
   - Título de la Evaluación
   - Materia/Asignatura (inferir del contenido)
   - Nivel educativo sugerido
   - Descripción pedagógica extensa
   - Objetivos de aprendizaje evaluados
   - Palabras clave/Conceptos principales

2. COMPETENCIAS A EVALUAR
   - Competencias cognitivas (saber)
   - Competencias procedimentales (saber hacer)
   - Competencias actitudinales (saber ser)
   - Prerrequisitos necesarios

3. MATRIZ DE EVALUACIÓN EXTENSA
   Debes desglosar la evaluación en DIMENSIONES y CRITERIOS específicos.
   Para cada criterio, define:
   - **Nombre del Indicador** (Claro y unívoco)
   - **Descripción pedagógica**: Explicación del qué y el para qué.
   - **Escala de Valoración Detallada**:
     * Excelente (4): Descripción del desempeño sobresaliente.
     * Bueno (3): Descripción del desempeño adecuado.
     * Suficiente (2): Descripción del desempeño mínimo aceptable.
     * Insuficiente (1): Descripción de la carencia o error.
   - **Peso relativo** (%)
   - **Evidencias requeridas**: Qué debe demostrar el estudiante.

4. NIVELES DE DOMINIO
   Describe los niveles de dominio esperados:
   - Nivel Básico: Qué debe saber/hacer mínimamente
   - Nivel Intermedio: Qué habilidades adicionales se esperan
   - Nivel Avanzado: Qué distingue a un estudiante sobresaliente

5. RECOMENDACIONES DE APLICACIÓN
   - Contexto de uso sugerido (examen, proyecto, práctica)
   - Tiempo estimado de aplicación
   - Sugerencias de retroalimentación
   - Criterios de aprobación

REGLAS DE GENERACIÓN:
- SÉ EXTENSO: No seas sucinto. Explica cada punto detalladamente.
- Usa lenguaje académico formal y preciso.
- Relaciona los criterios con los CONCEPTOS del contenido proporcionado.
- Si faltan datos en el contexto, infiere las mejores prácticas pedagógicas estándar.
- El formato final debe ser TEXTO PLANO legible, usando Markdown para estructura (#, ##, **negrita**, tablas si es apropiado).
- ADAPTA la rúbrica a la materia específica inferida del contenido.

GENERA AHORA LA RÚBRICA COMPLETA Y EXTENSA:
"""

        def hacer_llamada():
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_generacion,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=60000,
                )
            )
            
            # Verificar si la respuesta fue truncada
            if response.candidates and response.candidates[0].finish_reason != "STOP":
                print(f"⚠️ Advertencia: Respuesta finalizada por {response.candidates[0].finish_reason}")
                if response.candidates[0].finish_reason == "MAX_TOKENS":
                    print("⚠️ La rúbrica ha sido truncada por exceder el límite de tokens.")
            
            return response.text

        try:
            resultado = llamar_llm_con_retry(hacer_llamada, max_intentos=3, backoff_base=10)

            if resultado:
                return resultado
            else:
                # Si falla por rate limit, generar rúbrica básica desde el contexto
                return self._generar_rubrica_basica(prompt, info_busqueda)

        except Exception as e:
            print(f"⚠️ Error generando rúbrica: {e}")
            return self._generar_rubrica_basica(prompt, info_busqueda)

    def _generar_rubrica_basica(self, prompt: str, info_busqueda: Dict) -> str:
        """Genera una rúbrica básica sin LLM cuando hay problemas de rate limit"""
        print("📝 Generando rúbrica básica (sin LLM)...")

        resultados = info_busqueda.get('resultados', [])

        rubrica = f"""
================================================================================
RÚBRICA DE EVALUACIÓN ACADÉMICA
================================================================================

Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Solicitud: {prompt}

--------------------------------------------------------------------------------
CRITERIOS DE EVALUACIÓN
--------------------------------------------------------------------------------

"""

        for i, resultado in enumerate(resultados[:15], 1):
            nombre = resultado.get('nombre', 'Sin nombre')
            tipo = resultado.get('tipo', 'criterio')
            contexto = resultado.get('contexto', '')[:200]

            rubrica += f"""
{i}. {nombre.replace('_', ' ').title()}
   Tipo: {tipo}
   Descripción: {contexto}

   □ Cumple    □ No cumple    □ No aplica

   Observaciones: ___________________________________________________________
   _________________________________________________________________________

"""

        rubrica += f"""
================================================================================
VALIDACIÓN Y FIRMA
================================================================================

Evaluador: _________________________    Fecha: _____________________________

Firma: _____________________________

================================================================================
"""

        return rubrica


# ============================================================================
# SISTEMA COLABA PRINCIPAL (MEJORADO)
# ============================================================================

class SistemaColaba:
    """Sistema principal con ontología dinámica"""

    def __init__(self):
        print("\n" + "="*80)
        print("🎓 SISTEMA COLABA v3.0 - ONTOLOGÍA DINÁMICA".center(80))
        print("="*80 + "\n")

        # Configuración
        self.config = ConfiguracionColaba()

        # Inicializar componentes base
        print("💾 Inicializando Agente Persistencia...")
        self.agente_persistencia = AgentePersistencia(self.config)

        # Inicializar ontología dinámica
        print("🧠 Inicializando Ontología Dinámica...")
        self.ontologia_dinamica = OntologiaDinamica(
            self.agente_persistencia.driver,
            genai.Client(api_key=self.config.GOOGLE_API_KEY)
        )

        # Cargar schema inicial
        self.ontologia_dinamica.cargar_schema_actual()

        # Inicializar resolución de entidades
        print("🔗 Inicializando Resolución de Entidades...")
        self.resolucion_entidades = ResolucionEntidades(
            self.agente_persistencia.driver,
            self.agente_persistencia.embedding_model,
            genai.Client(api_key=self.config.GOOGLE_API_KEY)
        )

        # Inicializar validador ontológico
        print("✅ Inicializando Validador Ontológico...")
        self.validador_ontologico = AgenteValidadorOntologico(
            self.config,
            self.ontologia_dinamica,
            self.resolucion_entidades
        )

        # Inicializar validador de consistencia
        print("📊 Inicializando Validador de Consistencia...")
        self.validador_consistencia = ValidadorConsistencia(
            self.agente_persistencia.driver
        )

        # Inicializar agentes principales
        print("🤖 Inicializando Agente Ontólogo (Schema-Aware)...")
        self.agente_ontologo = AgenteOntologo(self.config, self.ontologia_dinamica)

        print("🔍 Inicializando Agente Búsqueda...")
        self.agente_busqueda = AgenteBusqueda(self.config, self.agente_persistencia)

        print("📋 Inicializando Agente Rubricador...")
        self.agente_rubricador = AgenteRubricador(self.config)

        print("\n" + "="*80)
        print("✓ SISTEMA COLABA V3.0 LISTO".center(80))
        print("="*80 + "\n")

    def cargar_documento_normativo(self, documento: str, tipo: str) -> Dict:
        """Pipeline mejorado: Documento → Ontología → Validación → Neo4j"""

        # 1. Extraer ontología (schema-aware)
        ontologia = self.agente_ontologo.analizar_documento(documento, tipo)

        # 2. Validar y normalizar entidades
        ontologia_validada, resultados_validacion = self.validador_ontologico.validar_ontologia(ontologia)

        # 3. Inferir relaciones adicionales
        print("\n🔍 Infiriendo relaciones adicionales...")
        relaciones_inferidas = self.ontologia_dinamica.inferir_relaciones_posibles(
            ontologia_validada.entidades
        )

        if relaciones_inferidas:
            print(f"   Relaciones inferidas: {len(relaciones_inferidas)}")
            ontologia_validada.relaciones.extend(relaciones_inferidas)

        # 4. Actualizar schema dinámico
        print("\n📋 Actualizando schema ontológico...")
        self.ontologia_dinamica.actualizar_schema(
            ontologia_validada.entidades,
            ontologia_validada.relaciones
        )

        # 5. Persistir en Neo4j
        resultado_persistencia = self.agente_persistencia.guardar_ontologia(ontologia_validada)

        return {
            "ontologia_original": ontologia,
            "ontologia_validada": ontologia_validada,
            "validacion": {
                "total_validaciones": len(resultados_validacion),
                "entidades_fusionadas": sum(1 for r in resultados_validacion if r.entidades_fusionadas),
                "entidades_nuevas": sum(1 for r in resultados_validacion if r.es_nueva)
            },
            "persistencia": resultado_persistencia
        }

    def generar_rubrica(self, prompt_usuario: str, archivo_salida: str = None) -> str:
        """Pipeline: Prompt → Búsqueda → RAG → Rúbrica"""

        contexto = self.agente_busqueda.procesar_prompt(prompt_usuario)
        rubrica = self.agente_rubricador.generar_rubrica(prompt_usuario, contexto)

        if archivo_salida:
            with open(archivo_salida, 'w', encoding='utf-8') as f:
                f.write(rubrica)
                f.flush()
                os.fsync(f.fileno())
            
            size = os.path.getsize(archivo_salida)
            print(f"\n💾 Rúbrica guardada en: {archivo_salida} ({size/1024:.1f} KB)")

        return rubrica

    def generar_reporte_calidad(self) -> str:
        """Genera reporte de calidad ontológica"""
        print("\n📊 Generando reporte de calidad ontológica...")

        metricas = self.validador_consistencia.calcular_metricas()
        reporte = self.validador_consistencia.generar_reporte_calidad(metricas)
        anomalias = self.validador_consistencia.detectar_anomalias(metricas)

        if anomalias:
            reporte += "\n⚠️  ANOMALÍAS DETECTADAS:\n"
            for anomalia in anomalias:
                reporte += f"{anomalia}\n"
        else:
            reporte += "\n✓ No se detectaron anomalías significativas\n"

        return reporte

    def obtener_estadisticas_schema(self) -> Dict:
        """Obtiene estadísticas del schema actual"""
        schema = self.ontologia_dinamica.cargar_schema_actual()

        return {
            "tipos_entidad": len(schema.tipos_entidad),
            "tipos_relacion": len(schema.tipos_relacion),
            "total_entidades": schema.estadisticas.get('total_entidades', 0),
            "total_relaciones": schema.estadisticas.get('total_relaciones', 0),
            "ultima_actualizacion": self.ontologia_dinamica.ultima_actualizacion.isoformat()
                if self.ontologia_dinamica.ultima_actualizacion else None,
            "detalle_tipos": schema.tipos_entidad
        }

    def obtener_estadisticas_api(self) -> Dict:
        """Obtiene estadísticas de uso de la API de Gemini (rate limiter y caché)"""
        return {
            'rate_limiter': rate_limiter.get_stats(),
            'cache': llm_cache.get_stats()
        }

    def close(self):
        """Cierra conexiones y muestra estadísticas de uso"""
        # Mostrar estadísticas de uso de API
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS DE USO DE API")
        print("="*80)
        
        stats = self.obtener_estadisticas_api()
        
        print(f"\n🔄 Rate Limiter:")
        print(f"   - Total llamadas API: {stats['rate_limiter']['total_llamadas']}")
        print(f"   - Intervalo mínimo: {stats['rate_limiter']['intervalo_minimo']}s")
        
        print(f"\n📦 Caché LLM:")
        print(f"   - Cache hits: {stats['cache']['hits']}")
        print(f"   - Cache misses: {stats['cache']['misses']}")
        print(f"   - Hit rate: {stats['cache']['hit_rate']}")
        print(f"   - Tamaño caché: {stats['cache']['cache_size']} entradas")
        
        # Calcular ahorro estimado
        if stats['cache']['hits'] > 0:
            tiempo_ahorrado = stats['cache']['hits'] * 12  # 12s por llamada ahorrada
            print(f"\n⚡ Tiempo ahorrado por caché: ~{tiempo_ahorrado}s ({tiempo_ahorrado/60:.1f} min)")
        
        print("="*80 + "\n")
        
        self.agente_persistencia.close()


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Inicializar sistema
    colaba = SistemaColaba()

    # ========================================================================
    # ========================================================================
    # NORMATIVA 1: CALIDAD DE APUNTES DE CÁTEDRA
    # ========================================================================
    normativa_apuntes = """
    NORMATIVA DE CALIDAD PARA LA ELABORACIÓN DE APUNTES DE CÁTEDRA

    ARTÍCULO 1: DESARROLLO DE CONCEPTOS
    Los apuntes deben presentar el contenido disciplinar con rigor académico y claridad expositiva.
    Es fundamental que la estructura lógica facilite la comprensión y el estudio autónomo.
    Criterios de evaluación:
    - Precisión conceptual: Definiciones claras, unívocas y técnicamente correctas.
    - Profundidad adecuada: El nivel de detalle corresponde a los objetivos de aprendizaje.
    - Secuenciación lógica: Progresión coherente de ideas, de lo simple a lo complejo.
    - Ejemplificación: Uso de ejemplos relevantes y contextualizados para ilustrar conceptos abstractos.
    - Síntesis y análisis: Capacidad de integrar información de diversas fuentes y aportar valor añadido.

    ARTÍCULO 2: REFERENCIAS BIBLIOGRÁFICAS
    Todo material docente debe estar fundamentado en fuentes académicas reconocidas y actualizadas.
    La honestidad intelectual y el rigor metodológico son imperativos.
    Criterios de evaluación:
    - Citación correcta: Uso consistente de un estilo de citación estándar (APA, ISO 690, IEEE).
    - Pertinencia: Selección de bibliografía relevante, actualizada y de calidad académica.
    - Distinción de fuentes: Claridad entre bibliografía básica (obligatoria) y complementaria (ampliación).
    - Variedad de fuentes: Inclusión de manuales, artículos científicos y recursos especializados.

    ARTÍCULO 3: RECURSOS Y ENLACES WEB
    Los recursos digitales complementarios deben enriquecer el aprendizaje y estar seleccionados criteriosamente.
    La accesibilidad y la permanencia de la información son aspectos clave.
    Criterios de evaluación:
    - Validez de enlaces: Todos los hipervínculos deben estar activos y dirigir al recurso correcto.
    - Calidad de recursos: Selección de sitios web confiables, institucionales o académicos.
    - Descripción de contenido: Cada enlace debe acompañarse de una breve descripción de su utilidad.
    - Accesibilidad: Preferencia por recursos de acceso abierto o disponibles a través de la biblioteca institucional.
    - Contextualización: Explicación de cómo el recurso digital se integra con los contenidos teóricos.
    """

    print("📚 Cargando Normativa de Calidad de Apuntes de Cátedra...")
    resultado_apuntes = colaba.cargar_documento_normativo(
        documento=normativa_apuntes,
        tipo="normativa_calidad_apuntes"
    )

    print(f"\n{'='*80}")
    print("RESULTADOS - NORMATIVA CALIDAD APUNTES")
    print(f"{'='*80}")
    print(f"Entidades originales: {len(resultado_apuntes['ontologia_original'].entidades)}")
    print(f"Entidades validadas: {len(resultado_apuntes['ontologia_validada'].entidades)}")
    print(f"Entidades fusionadas: {resultado_apuntes['validacion']['entidades_fusionadas']}")
    print(f"Entidades nuevas: {resultado_apuntes['validacion']['entidades_nuevas']}")

    # ========================================================================
    # ========================================================================
    # NORMATIVA 2: IEEE SOFTWARE ENGINEERING EDUCATION (OPCIONAL - COMENTADO)
    # ========================================================================
    # (Comentado para priorizar el caso de uso actual: Apuntes de Cátedra)
    """
    normativa_ieee_see = ... 
    """

    # ========================================================================
    # NORMATIVA 3: IEEE LOM (Learning Object Metadata)
    # ========================================================================
    documento_lom = """
    Ontología de Aprendizaje (Learning Ontology)

    Descripción General:
    Se utiliza para representar conceptos relacionados con el aprendizaje, como objetivos de aprendizaje, actividades, recursos y evaluaciones.
    Esta ontología se basa en el estándar IEEE LOM (Learning Object Metadata) que define un conjunto de metadatos para describir objetos de aprendizaje.

    CATEGORÍA 1: GENERAL
    Agrupa la información general que describe el objeto de aprendizaje como un todo.
    - Identificador: Etiqueta única global.
    - Título: Nombre del objeto de aprendizaje.
    - Idioma: Lengua principal utilizada.
    - Descripción: Descripción textual del contenido.
    - Palabras clave: Términos descriptivos.
    - Cobertura: Ámbito geográfico o temporal.
    - Estructura: Atómica, colección, lineal, jerárquica.
    - Nivel de Agregación: 1 (raw media), 2 (lesson), 3 (course), 4 (certificate).

    CATEGORÍA 2: CICLO DE VIDA
    Historia y estado actual del objeto de aprendizaje.
    - Versión: Edición del objeto.
    - Estado: Borrador, Final, Revisado, Obsoleto.
    - Contribución: Entidades (personas, organizaciones) que han contribuido (Autor, Editor, Validador, Traductor).

    CATEGORÍA 3: META-METADATOS
    Información sobre el propio registro de metadatos.
    - Identificador y Esquema de Metadatos.
    - Lenguaje de los metadatos.

    CATEGORÍA 4: TÉCNICA
    Requisitos técnicos y características.
    - Formato: Tipo MIME (text/html, video/mp4).
    - Tamaño: En bytes.
    - Ubicación: URL o URI.
    - Requerimientos: Hardware y software necesario.
    - Duración: Tiempo de interacción.

    CATEGORÍA 5: EDUCACIONAL
    Características pedagógicas y educativas.
    - Tipo de Interactividad: Activa, Expositiva, Mixta.
    - Tipo de Recurso de Aprendizaje: Ejercicio, Simulación, Cuestionario, Rúbrica, Figura, Gráfico, Diapositiva.
    - Nivel de Interactividad: Muy bajo a Muy alto.
    - Densidad Semántica: Grado de concisión.
    - Usuario Final Previsto: Docente, Alumno, Gestor.
    - Contexto de Aprendizaje: Ed. Primaria, Secundaria, Universidad, Formación Continua.
    - Rango de Edad Típico.
    - Dificultad: Muy fácil a Muy difícil.
    - Tiempo de Aprendizaje Típico.

    CATEGORÍA 6: DERECHOS
    Condiciones de uso y propiedad intelectual.
    - Costo: Sí/No.
    - Derechos de Autor y otras restricciones.
    - Descripción de los derechos.

    CATEGORÍA 7: RELACIÓN
    Relaciones con otros objetos de aprendizaje.
    - Tipo de Relación: Es parte de, Tiene parte, Es versión de, Es formato de, Referencia a, Se basa en.
    - Recurso: Identificador del recurso relacionado.

    CATEGORÍA 8: ANOTACIÓN
    Comentarios sobre el uso educativo del objeto.
    - Entidad: Quién hace la anotación.
    - Fecha.
    - Descripción.

    CATEGORÍA 9: CLASIFICACIÓN
    Describe el objeto en relación con un sistema de clasificación particular (taxonomía).
    - Propósito: Disciplina, Idea educativa, Nivel de habilidad.
    - Ruta Taxonómica: Fuente (ej. Dewey, UNESCO) y Taxón (código/entrada).
    """

    # Cargar documento IEEE LOM
    print("\n📚 Cargando documento normativo IEEE LOM...")
    resultado_lom = colaba.cargar_documento_normativo(
        documento=documento_lom,
        tipo="normativa_ieee_lom"
    )

    print(f"\n{'='*80}")
    print("RESULTADOS - NORMATIVA IEEE LOM")
    print(f"{'='*80}")
    print(f"Entidades originales: {len(resultado_lom['ontologia_original'].entidades)}")
    print(f"Entidades validadas: {len(resultado_lom['ontologia_validada'].entidades)}")
    print(f"Entidades fusionadas: {resultado_lom['validacion']['entidades_fusionadas']}")
    print(f"Entidades nuevas: {resultado_lom['validacion']['entidades_nuevas']}")

    # Generar reporte de calidad
    print("\n" + "="*80)
    print("REPORTE DE CALIDAD ONTOLÓGICA (TODAS LAS NORMATIVAS)")
    print("="*80)
    reporte = colaba.generar_reporte_calidad()
    print(reporte)

    # Mostrar estadísticas del schema
    print("\n" + "="*80)
    print("ESTADÍSTICAS DEL SCHEMA")
    print("="*80)
    stats = colaba.obtener_estadisticas_schema()
    print(f"Tipos de entidad: {stats['tipos_entidad']}")
    print(f"Tipos de relación: {stats['tipos_relacion']}")
    print(f"Total entidades: {stats['total_entidades']}")
    print(f"Total relaciones: {stats['total_relaciones']}")

    # Generar rúbrica para evaluar calidad de apuntes de cátedra
    print("\n📋 Generando rúbrica de evaluación de APUNTES DE CÁTEDRA...")
    rubrica = colaba.generar_rubrica(
        prompt_usuario="""Genera una rúbrica detallada para evaluar la CALIDAD DE APUNTES DE CÁTEDRA
        basada en la NORMATIVA DE CALIDAD PARA LA ELABORACIÓN DE APUNTES.

        La rúbrica debe evaluar exhaustivamente:

        1. DESARROLLO CONCEPTUAL:
           - Claridad y precisión en las definiciones
           - Profundidad y rigor académico
           - Estructura lógica y secuenciación de contenidos
           - Calidad de los ejemplos y casos prácticos

        2. REFERENCIAS Y BIBLIOGRAFÍA:
           - Uso correcto de normas de citación (APA, ISO 690)
           - Actualidad y relevancia de fuentes consultadas
           - Distinción clara entre bibliografía básica y complementaria

        3. RECURSOS DIGITALES Y ENLACES:
           - Validez y funcionamiento de todos los enlaces
           - Calidad y confiabilidad de los sitios web referenciados
           - Pertinencia pedagógica de los recursos externos
           - Accesibilidad de los materiales enlazados

        Cada criterio debe tener niveles: EXCELENTE, BUENO, SUFICIENTE, INSUFICIENTE.
        Incluir campo de observaciones y recomendaciones de mejora.""",
        archivo_salida="rubrica_calidad_apuntes.txt"
    )

    print("\n" + "="*80)
    print("RÚBRICA DE EVALUACIÓN - CALIDAD DE APUNTES DE CÁTEDRA")
    print("="*80)
    print(rubrica[:2000] + "...")  # Mostrar más del contenido

    # Cerrar
    colaba.close()
    print("\n✓ Sistema cerrado correctamente")