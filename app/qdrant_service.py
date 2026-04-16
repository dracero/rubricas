"""
Unified Qdrant Service + ADK Tool Functions.

Consolidates the QdrantService that was previously duplicated across
generator, evaluator, and corrector agents into a single module.
Uses OpenAI text-embedding-3-small for embeddings (1536 dimensions) via LiteLLM.
"""

import os
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import litellm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    QDRANT_AVAILABLE = False

from common.config import ConfiguracionColaba, traceable, get_current_run_tree
from app.domain import (
    Entidad, Relacion, Ontologia,
    parsear_json_con_fallback,
)

logger = logging.getLogger(__name__)

# OpenAI embedding config (via LiteLLM)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
VERSION = "OPENAI_EMBED_1536_V1"

# ============================================================================
# Module-level singleton
# ============================================================================
_qdrant_service: Optional["QdrantService"] = None


def _get_qdrant_service() -> "QdrantService":
    """Get or create the module-level QdrantService singleton."""
    global _qdrant_service
    if _qdrant_service is None:
        config = ConfiguracionColaba()
        _qdrant_service = QdrantService(config)
    return _qdrant_service


# ============================================================================
# QDRANT SERVICE
# ============================================================================

class QdrantService:
    """Unified Qdrant vector DB service: embeddings, save, search."""

    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.collection_name = "rubricas_entidades"

        # Qdrant client — respects VECTOR_MODE env var
        vector_mode = os.getenv("VECTOR_MODE", "server").lower()

        if not QDRANT_AVAILABLE:
            self.client = None
            logger.warning("⚠️ qdrant-client no instalado")
        elif vector_mode == "memory":
            self.client = QdrantClient(location=":memory:")
            logger.info("Qdrant: modo memoria (efímero)")
        elif vector_mode == "disk":
            disk_path = os.getenv("QDRANT_DISK_PATH", "./qdrant_data")
            self.client = QdrantClient(path=disk_path)
            logger.info("Qdrant: modo disco (%s)", disk_path)
        elif config.QDRANT_URL:
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
            )
            logger.info("Qdrant: modo server (%s)", config.QDRANT_URL)
        else:
            self.client = None
            logger.warning("⚠️ Sin conexión a Qdrant (VECTOR_MODE=%s, sin URL)", vector_mode)

        logger.info(f"✅ OpenAI embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION}d) [VERSION: {VERSION}]")

        self._init_collection()

    def _init_collection(self):
        """Create the Qdrant collection if it doesn't exist, or recreate if dimensions mismatch."""
        if not self.client:
            return
        try:
            collections = self.client.get_collections()
            exists = any(
                c.name == self.collection_name
                for c in collections.collections
            )
            
            if exists:
                # Check dimensions
                info = self.client.get_collection(self.collection_name)
                current_dim = info.config.params.vectors.size
                if current_dim != EMBEDDING_DIMENSION:
                    logger.warning(
                        f"⚠️ Dimension mismatch in {self.collection_name}: "
                        f"current={current_dim}, expected={EMBEDDING_DIMENSION}. "
                        "Recreating collection..."
                    )
                    self.client.delete_collection(self.collection_name)
                    exists = False

            if not exists:
                logger.info(f"📦 Creating Qdrant collection: {self.collection_name} (dim={EMBEDDING_DIMENSION})")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"⚠️ Error initializing Qdrant: {e}")

    def clear_collection(self):
        """Delete and recreate the Qdrant collection."""
        if not self.client:
            logger.warning("⚠️ No Qdrant client — skipping clear")
            return False
        try:
            collections = self.client.get_collections()
            exists = any(
                c.name == self.collection_name
                for c in collections.collections
            )
            if exists:
                logger.info(f"🗑️ Deleting Qdrant collection: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)

            logger.info(f"📦 Recreating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE
                )
            )
            logger.info("✅ Qdrant collection cleared and recreated")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing Qdrant collection: {e}")
            return False

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text using OpenAI text-embedding-3-small via LiteLLM."""
        response = litellm.embedding(model=EMBEDDING_MODEL, input=[text])
        return response.data[0]["embedding"]

    # ------------------------------------------------------------------ #
    # Generic collection methods (used by RubricRepositoryService, etc.)
    # ------------------------------------------------------------------ #

    def init_collection(self, collection_name: str) -> None:
        """Create a collection with 1536d cosine config if it doesn't already exist."""
        if not self.client:
            return
        try:
            collections = self.client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            if not exists:
                logger.info(f"📦 Creating Qdrant collection: {collection_name} (dim={EMBEDDING_DIMENSION})")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.error(f"⚠️ Error initializing collection '{collection_name}': {e}")

    def upsert_point(self, collection_name: str, point_id: str, vector: List[float], payload: Dict) -> bool:
        """Generic upsert of a single point to any collection. Returns True on success."""
        if not self.client:
            return False
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    qmodels.PointStruct(id=point_id, vector=vector, payload=payload),
                ],
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error upserting point to '{collection_name}': {e}")
            return False

    def search_collection(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """Embed *query_text*, search *collection_name*, return list of dicts with payload + score."""
        if not self.client:
            return []
        try:
            vector = self.embed(query_text)
            result = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            hits = result.points if hasattr(result, "points") else result
            results: List[Dict] = []
            for hit in hits:
                payload = hit.payload.copy() if hit.payload else {}
                payload["score"] = hit.score
                results.append(payload)
            return results
        except Exception as e:
            logger.error(f"⚠️ Error searching collection '{collection_name}': {e}")
            return []

    def get_point(self, collection_name: str, point_id: str) -> Optional[Dict]:
        """Retrieve a single point by ID. Returns the payload dict or None if not found."""
        if not self.client:
            return None
        try:
            points = self.client.retrieve(collection_name=collection_name, ids=[point_id])
            if not points:
                return None
            payload = points[0].payload.copy() if points[0].payload else {}
            return payload
        except Exception as e:
            logger.error(f"⚠️ Error retrieving point '{point_id}' from '{collection_name}': {e}")
            return None

    def scroll_collection(
        self,
        collection_name: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[Dict], int]:
        """Paginated listing of points in a collection.

        Returns ``(list_of_payload_dicts, total_count)``.
        Each dict includes a ``_point_id`` key with the point's ID.
        """
        if not self.client:
            return [], 0
        try:
            # Total count
            count_result = self.client.count(collection_name=collection_name)
            total = count_result.count

            # Scroll enough points to cover offset + limit, then slice
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=offset + limit,
            )
            sliced = points[offset: offset + limit]
            results: List[Dict] = []
            for point in sliced:
                payload = point.payload.copy() if point.payload else {}
                payload["_point_id"] = point.id
                results.append(payload)
            return results, total
        except Exception as e:
            logger.error(f"⚠️ Error scrolling collection '{collection_name}': {e}")
            return [], 0

    def delete_point(self, collection_name: str, point_id: str) -> bool:
        """Delete a point by ID. Returns True if deleted, False on error."""
        if not self.client:
            return False
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=qmodels.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting point '{point_id}' from '{collection_name}': {e}")
            return False

    @traceable(name="QdrantService.save_ontology", run_type="chain")
    def save_ontology(self, ontologia: Ontologia) -> bool:
        """Save ontology entities and relations to Qdrant."""
        if not self.client:
            logger.warning("⚠️ No Qdrant client — skipping save")
            return False

        points = []

        relations_by_entity = defaultdict(list)
        for rel in ontologia.relaciones:
            relations_by_entity[rel.origen].append(rel.to_dict())

        for entidad in ontologia.entidades:
            point_id = hashlib.md5(entidad.nombre.encode()).hexdigest()
            text_for_embedding = f"{entidad.nombre}: {entidad.contexto}"
            vector = self.embed(text_for_embedding)

            payload = entidad.to_dict()
            payload["relaciones_salientes"] = relations_by_entity[entidad.nombre]

            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        if points:
            try:
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.extra = run_tree.extra or {}
                    run_tree.extra.update({
                        "qdrant_operation": "upsert",
                        "collection_name": self.collection_name,
                        "num_entities": len(ontologia.entidades),
                        "num_relations": len(ontologia.relaciones),
                        "num_points": len(points),
                        "vector_dimension": EMBEDDING_DIMENSION,
                    })

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"✅ Saved {len(points)} entities to Qdrant")
                return True
            except Exception as e:
                logger.error(f"❌ Error saving to Qdrant: {e}")
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.error = str(e)
                return False
        return False

    @traceable(name="QdrantService.save_ontology_additive", run_type="chain")
    def save_ontology_additive(self, ontologia: Ontologia, source_document_id: str, source_filename: str) -> bool:
        """Save ontology to Qdrant additively (upsert) with source document tracking.

        Does NOT clear the collection — only adds/updates entities.
        Each entity payload includes source_document_id and source_filename
        for traceability within a batch.
        """
        if not self.client:
            logger.warning("⚠️ No Qdrant client — skipping additive save")
            return False

        points = []

        relations_by_entity = defaultdict(list)
        for rel in ontologia.relaciones:
            relations_by_entity[rel.origen].append(rel.to_dict())

        for entidad in ontologia.entidades:
            point_id = hashlib.md5(entidad.nombre.encode()).hexdigest()
            text_for_embedding = f"{entidad.nombre}: {entidad.contexto}"
            vector = self.embed(text_for_embedding)

            payload = entidad.to_dict()
            payload["relaciones_salientes"] = relations_by_entity[entidad.nombre]
            payload["source_document_id"] = source_document_id
            payload["source_filename"] = source_filename

            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        if points:
            try:
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.extra = run_tree.extra or {}
                    run_tree.extra.update({
                        "qdrant_operation": "upsert_additive",
                        "collection_name": self.collection_name,
                        "source_document_id": source_document_id,
                        "source_filename": source_filename,
                        "num_entities": len(ontologia.entidades),
                        "num_relations": len(ontologia.relaciones),
                        "num_points": len(points),
                        "vector_dimension": EMBEDDING_DIMENSION,
                    })

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(
                    f"✅ Additive save: {len(points)} entities from "
                    f"'{source_filename}' (doc_id={source_document_id})"
                )
                return True
            except Exception as e:
                logger.error(f"❌ Error in additive save to Qdrant: {e}")
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.error = str(e)
                return False
        return False

    @traceable(name="QdrantService.search", run_type="retriever")
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.5) -> List[Dict]:
        """Search for similar entities by vector."""
        if not self.client:
            return []

        vector = self.embed(query)

        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                score_threshold=score_threshold
            )
            hits = result.points if hasattr(result, 'points') else result

            resultados = []
            for hit in hits:
                payload = hit.payload.copy() if hit.payload else {}
                payload['score'] = hit.score
                resultados.append(payload)

            avg_score = (
                sum(r['score'] for r in resultados) / len(resultados)
                if resultados else 0
            )

            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.extra = run_tree.extra or {}
                run_tree.extra.update({
                    "qdrant_operation": "search",
                    "collection_name": self.collection_name,
                    "query": query[:100],
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "num_results": len(resultados),
                    "avg_score": round(avg_score, 3),
                    "top_scores": [round(r['score'], 3) for r in resultados[:5]],
                })

            logger.info(
                f"📊 Qdrant search: {len(resultados)} hits, "
                f"avg_score: {avg_score:.3f}"
            )
            return resultados
        except Exception as e:
            logger.error(f"⚠️ Qdrant search error: {e}")
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.error = str(e)
            return []


# ============================================================================
# ADK TOOL FUNCTIONS (shared by all skills)
# ============================================================================

def guardar_ontologia_en_qdrant(ontologia_json: str) -> str:
    """Parses an ontology JSON and saves entities/relations to Qdrant.

    Use this tool after extracting entities and relations from a normative document.
    The JSON must have the structure:
    {
        "entidades": [{"nombre": "...", "tipo": "...", "contexto": "...", "propiedades": {...}}],
        "relaciones": [{"origen": "...", "destino": "...", "tipo": "...", "propiedades": {...}}]
    }

    Args:
        ontologia_json: JSON string with the extracted ontology (entidades + relaciones).

    Returns:
        A confirmation message with the number of entities and relations saved.
    """
    try:
        data = parsear_json_con_fallback(ontologia_json)

        entidades = []
        for e in data.get("entidades", []):
            entidades.append(Entidad(
                nombre=e.get("nombre", "Desconocido"),
                tipo=e.get("tipo", "Desconocido"),
                propiedades=e.get("propiedades", {}),
                contexto=e.get("contexto", ""),
                fecha_creacion=datetime.now().isoformat()
            ))

        relaciones = []
        for r in data.get("relaciones", []):
            relaciones.append(Relacion(
                origen=r.get("origen", "Desconocido"),
                destino=r.get("destino", "Desconocido"),
                tipo=r.get("tipo", "Desconocido"),
                propiedades=r.get("propiedades", {})
            ))

        ontologia = Ontologia(
            entidades=entidades,
            relaciones=relaciones,
            metadata={"source": "skill_ontologo"}
        )

        qdrant = _get_qdrant_service()
        
        # Clear existing collection before saving new ontology
        logger.info("🗑️ Limpiando colección Qdrant antes de guardar nueva ontología...")
        qdrant.clear_collection()
        
        success = qdrant.save_ontology(ontologia)

        if success:
            return (
                f"✅ Ontología guardada exitosamente en Qdrant: "
                f"{len(entidades)} entidades, {len(relaciones)} relaciones."
            )
        else:
            return "⚠️ No se pudieron guardar entidades en Qdrant."

    except Exception as e:
        logger.error(f"Error in guardar_ontologia_en_qdrant: {e}")
        return f"❌ Error guardando ontología: {str(e)}"


def buscar_contexto_qdrant(consulta: str) -> str:
    """Searches the Qdrant vector database for normative context relevant to the query.

    Use this tool to retrieve knowledge from previously indexed normative documents.
    Returns entities with their similarity scores, descriptions, and relationships.

    Args:
        consulta: The search query describing what normative context is needed.

    Returns:
        A formatted string with the relevant normative context found in Qdrant.
    """
    try:
        qdrant = _get_qdrant_service()
        results = qdrant.search(consulta, limit=10, score_threshold=0.4)

        if not results:
            return "No se encontró contexto normativo relevante en la base de conocimiento."

        lines = [f"📚 Contexto normativo encontrado ({len(results)} documentos):\n"]
        for item in results:
            score = item.get('score', 0)
            nombre = item.get('nombre', 'N/A')
            contexto = item.get('contexto', '')[:300]
            lines.append(f"- [{score:.3f}] **{nombre}**: {contexto}")

            for rel in item.get('relaciones_salientes', []):
                lines.append(
                    f"  → {rel.get('tipo', '?')} → {rel.get('destino', '?')}"
                )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in buscar_contexto_qdrant: {e}")
        return f"❌ Error buscando contexto: {str(e)}"

def leer_rubrica_subida(rubric_id: str) -> str:
    """Reads the content of a rubric file that was previously uploaded via the frontend.

    Use this tool when you need to access the text of a rubric that the user
    uploaded through the UI. The rubric_id is provided by the frontend after upload.

    Args:
        rubric_id: The unique ID of the uploaded rubric (provided after upload).

    Returns:
        The text content of the rubric file, or an error message if not found.
    """
    from pathlib import Path

    upload_dir = Path("/tmp/rubricas_uploads")

    # Try common extensions
    for ext in [".txt", ".md"]:
        rubric_path = upload_dir / f"rubric_{rubric_id}{ext}"
        if rubric_path.exists():
            try:
                text = rubric_path.read_text(encoding="utf-8")
                logger.info(f"📖 Rubric read: rubric_{rubric_id}{ext} ({len(text)} chars)")
                return text
            except Exception as e:
                return f"❌ Error leyendo rúbrica: {str(e)}"

    return f"❌ Rúbrica no encontrada con ID: {rubric_id}"


def leer_documento_subido(document_id: str) -> str:
    """Reads and extracts text from a PDF document that was previously uploaded via the frontend.

    Use this tool when you need to access the content of a document that the user
    uploaded for evaluation or analysis. The document_id is provided by the frontend after upload.
    Supports PDF files — text is automatically extracted from all pages.

    Args:
        document_id: The unique ID of the uploaded document (provided after upload).

    Returns:
        The extracted text content of the document, or an error message if not found.
    """
    from pathlib import Path

    upload_dir = Path("/tmp/rubricas_uploads")

    # Check both naming patterns: {id}.pdf and doc_{id}.pdf
    for pattern in [f"doc_{document_id}.pdf", f"{document_id}.pdf"]:
        doc_path = upload_dir / pattern
        if doc_path.exists():
            try:
                import pypdf
                reader = pypdf.PdfReader(str(doc_path))
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                full_text = "\n".join(text_parts)
                logger.info(f"📖 Document read: {pattern} ({len(full_text)} chars)")
                return full_text if full_text.strip() else "⚠️ El PDF no contiene texto extraíble."
            except ImportError:
                return "❌ pypdf no está instalado. No se puede leer el PDF."
            except Exception as e:
                return f"❌ Error leyendo documento: {str(e)}"

    return f"❌ Documento no encontrado con ID: {document_id}"


def buscar_rubricas_repositorio(consulta: str) -> str:
    """Busca rúbricas similares en el repositorio por consulta semántica.

    Usa esta herramienta para encontrar rúbricas previamente generadas que sean
    similares a un tema o consulta. Retorna una lista con scores de similitud,
    fechas, niveles y resúmenes.

    Args:
        consulta: Texto de búsqueda describiendo el tema o tipo de rúbrica buscada.

    Returns:
        Una cadena formateada con las rúbricas encontradas y sus metadatos.
    """
    try:
        from app.rubric_repository import _get_rubric_repository_service
        results = _get_rubric_repository_service().search_similar(consulta)

        if not results:
            return "No se encontraron rúbricas similares en el repositorio."

        lines = [f"📚 Rúbricas similares encontradas ({len(results)}):\n"]
        for item in results:
            score = item.get("score", 0)
            rubric_id = item.get("rubric_id", "N/A")
            level = item.get("level", "N/A")
            created_at = item.get("created_at", "N/A")
            summary = item.get("summary", "")[:200]
            filenames = ", ".join(item.get("source_filenames", []))
            lines.append(
                f"- [{score:.0%}] ID: {rubric_id}\n"
                f"  Nivel: {level} | Fecha: {created_at}\n"
                f"  Documentos fuente: {filenames}\n"
                f"  Resumen: {summary}\n"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in buscar_rubricas_repositorio: {e}")
        return f"❌ Error buscando rúbricas: {str(e)}"


def obtener_rubrica_completa(rubric_id: str) -> str:
    """Recupera el texto completo y metadatos de una rúbrica del repositorio.

    Usa esta herramienta cuando necesites ver el contenido completo de una rúbrica
    específica del repositorio. Requiere el ID de la rúbrica.

    Args:
        rubric_id: El ID único de la rúbrica a recuperar.

    Returns:
        El texto completo de la rúbrica con sus metadatos, o un mensaje de error.
    """
    try:
        from app.rubric_repository import _get_rubric_repository_service
        rubric = _get_rubric_repository_service().get_rubric(rubric_id)

        if not rubric:
            return f"❌ Rúbrica no encontrada con ID: {rubric_id}"

        level = rubric.get("level", "N/A")
        created_at = rubric.get("created_at", "N/A")
        filenames = ", ".join(rubric.get("source_filenames", []))
        rubric_text = rubric.get("rubric_text", "Sin texto disponible")

        return (
            f"📄 Rúbrica: {rubric_id}\n"
            f"Nivel: {level} | Fecha: {created_at}\n"
            f"Documentos fuente: {filenames}\n\n"
            f"--- TEXTO COMPLETO ---\n{rubric_text}"
        )

    except Exception as e:
        logger.error(f"Error in obtener_rubrica_completa: {e}")
        return f"❌ Error recuperando rúbrica: {str(e)}"


# Registry of all available tool functions (for skill_loader)
TOOL_REGISTRY: Dict[str, Any] = {
    "guardar_ontologia_en_qdrant": guardar_ontologia_en_qdrant,
    "buscar_contexto_qdrant": buscar_contexto_qdrant,
    "leer_rubrica_subida": leer_rubrica_subida,
    "leer_documento_subido": leer_documento_subido,
    "buscar_rubricas_repositorio": buscar_rubricas_repositorio,
    "obtener_rubrica_completa": obtener_rubrica_completa,
}
