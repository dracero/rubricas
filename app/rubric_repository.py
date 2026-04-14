"""
RubricRepositoryService — persistent rubric storage and semantic search.

Stores generated rubrics in a dedicated Qdrant collection
(`rubricas_repositorio`) with their embedding vectors and metadata.
Provides search, retrieval, listing, and deletion operations.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.qdrant_service import QdrantService, _get_qdrant_service

logger = logging.getLogger(__name__)


class RubricRepositoryService:
    """Encapsulates all rubric repository logic on top of QdrantService."""

    COLLECTION_NAME = "rubricas_repositorio"

    def __init__(self, qdrant_service: QdrantService):
        self.qdrant = qdrant_service
        self._init_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_collection(self) -> None:
        """Create the ``rubricas_repositorio`` collection if it doesn't exist."""
        self.qdrant.init_collection(self.COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_rubric(
        self,
        rubric_text: str,
        level: str,
        source_filenames: List[str],
        source_document_ids: List[str],
    ) -> Optional[str]:
        """Store a rubric in the repository with LLM-generated topic index.

        Returns the generated ``rubric_id`` (UUID) on success, or ``None``
        if the upsert fails.
        """
        rubric_id = str(uuid.uuid4())

        # Generate topic index using LLM
        topics = self._generate_topics(rubric_text)
        logger.info(f"📋 Generated topics for rubric: {topics}")

        # Use topics + summary for the embedding (better semantic matching)
        embed_text = f"Temas: {', '.join(topics)}. {rubric_text[:500]}"
        try:
            vector = self.qdrant.embed(embed_text)
        except Exception as e:
            logger.error(f"❌ Error generating embedding for rubric: {e}")
            return None

        payload: Dict[str, Any] = {
            "rubric_id": rubric_id,
            "rubric_text": rubric_text,
            "summary": rubric_text[:300],
            "topics": topics,
            "level": level,
            "source_filenames": source_filenames,
            "source_document_ids": source_document_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        success = self.qdrant.upsert_point(
            collection_name=self.COLLECTION_NAME,
            point_id=rubric_id,
            vector=vector,
            payload=payload,
        )
        if success:
            logger.info(f"✅ Rubric stored: {rubric_id} | Topics: {topics}")
            return rubric_id

        logger.error(f"❌ Failed to store rubric {rubric_id}")
        return None

    def _generate_topics(self, rubric_text: str) -> List[str]:
        """Use LLM to extract a list of topic keywords from the rubric text."""
        try:
            import litellm
            import json

            logger.info("🏷️ Calling LLM to generate topics...")
            response = litellm.completion(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Extrae los temas principales de esta rúbrica de cumplimiento normativo. "
                        "Responde SOLO con un array JSON de strings, máximo 8 temas. "
                        "Cada tema debe ser una frase corta de 2-4 palabras. "
                        "Ejemplo: [\"calidad académica\", \"evaluación docente\", \"gestión curricular\"]"
                    )},
                    {"role": "user", "content": rubric_text[:3000]},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content or "[]"
            logger.info(f"🏷️ LLM topics raw response: {raw[:200]}")
            # Clean markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            topics = json.loads(raw.strip())
            if isinstance(topics, list):
                return [str(t) for t in topics[:8]]
        except Exception as e:
            logger.warning(f"⚠️ Could not generate topics: {e}")
        return []

    def search_similar(
        self,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Search for similar rubrics by embedding.

        Returns a list of dicts with ``score``, metadata fields, and a
        ``preview`` key containing the first 500 characters of the rubric text.
        """
        results = self.qdrant.search_collection(
            collection_name=self.COLLECTION_NAME,
            query_text=query_text,
            limit=limit,
            score_threshold=score_threshold,
        )

        items: List[Dict[str, Any]] = []
        for r in results:
            items.append({
                "rubric_id": r.get("rubric_id", ""),
                "summary": r.get("summary", ""),
                "level": r.get("level", ""),
                "source_filenames": r.get("source_filenames", []),
                "created_at": r.get("created_at", ""),
                "score": r.get("score", 0.0),
                "preview": r.get("rubric_text", "")[:500],
            })
        return items

    def get_rubric(self, rubric_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a full rubric by ID. Returns payload dict or ``None``."""
        return self.qdrant.get_point(self.COLLECTION_NAME, rubric_id)

    def list_rubrics(
        self,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List rubrics with pagination.

        * Without *search*: returns items ordered by ``created_at`` descending.
        * With *search*: performs semantic search and returns items ordered by
          ``score`` descending.

        ``limit`` is clamped to ``[1, 100]``; ``offset`` is clamped to ``>= 0``.
        """
        # Clamp parameters
        if limit <= 0:
            limit = 20
        limit = min(limit, 100)
        offset = max(offset, 0)

        if search:
            # Semantic search mode — score descending
            results = self.qdrant.search_collection(
                collection_name=self.COLLECTION_NAME,
                query_text=search,
                limit=limit,
                score_threshold=0.0,
            )
            items = []
            for r in results:
                items.append({
                    "rubric_id": r.get("rubric_id", ""),
                    "summary": r.get("summary", ""),
                    "topics": r.get("topics", []),
                    "level": r.get("level", ""),
                    "source_filenames": r.get("source_filenames", []),
                    "created_at": r.get("created_at", ""),
                    "score": r.get("score", 0.0),
                })
            # Already sorted by score desc from Qdrant
            return items, len(items)

        # Scroll mode — date descending
        items, total = self.qdrant.scroll_collection(
            collection_name=self.COLLECTION_NAME,
            limit=limit,
            offset=offset,
        )

        rubrics = []
        for item in items:
            rubrics.append({
                "rubric_id": item.get("rubric_id", item.get("_point_id", "")),
                "summary": item.get("summary", ""),
                "topics": item.get("topics", []),
                "level": item.get("level", ""),
                "source_filenames": item.get("source_filenames", []),
                "created_at": item.get("created_at", ""),
            })

        # Sort by created_at descending
        rubrics.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return rubrics, total

    def delete_rubric(self, rubric_id: str) -> bool:
        """Delete a rubric by ID. Returns ``True`` if deleted."""
        return self.qdrant.delete_point(self.COLLECTION_NAME, rubric_id)


# ============================================================================
# Module-level singleton
# ============================================================================

_rubric_repository_service: Optional[RubricRepositoryService] = None


def _get_rubric_repository_service() -> RubricRepositoryService:
    """Get or create the module-level RubricRepositoryService singleton."""
    global _rubric_repository_service
    if _rubric_repository_service is None:
        qdrant = _get_qdrant_service()
        _rubric_repository_service = RubricRepositoryService(qdrant)
    return _rubric_repository_service
