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
        """Store a rubric in the repository.

        Returns the generated ``rubric_id`` (UUID) on success, or ``None``
        if the upsert fails.
        """
        rubric_id = str(uuid.uuid4())
        try:
            vector = self.qdrant.embed(rubric_text)
        except Exception as e:
            logger.error(f"❌ Error generating embedding for rubric: {e}")
            return None

        payload: Dict[str, Any] = {
            "rubric_id": rubric_id,
            "rubric_text": rubric_text,
            "summary": rubric_text[:300],
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
            logger.info(f"✅ Rubric stored: {rubric_id}")
            return rubric_id

        logger.error(f"❌ Failed to store rubric {rubric_id}")
        return None

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
