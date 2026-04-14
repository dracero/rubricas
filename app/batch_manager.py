"""
In-memory batch state management for multi-document ontology extraction.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class ExtractionStatus(str, Enum):
    PENDIENTE = "pendiente"
    EN_PROCESO = "en_proceso"
    COMPLETADO = "completado"
    ERROR = "error"


@dataclass
class DocumentExtractionState:
    id: str
    filename: str
    status: ExtractionStatus = ExtractionStatus.PENDIENTE
    entities_count: int = 0
    relations_count: int = 0
    error_message: str = ""
    references: list = field(default_factory=list)


@dataclass
class BatchState:
    batch_id: str
    documents: Dict[str, DocumentExtractionState] = field(default_factory=dict)
    created_at: str = ""


class BatchManager:
    """Manages batch upload state and extraction task tracking."""

    def __init__(self) -> None:
        self._batches: Dict[str, BatchState] = {}

    def create_batch(self, file_ids: List[str], filenames: List[str]) -> str:
        """Create a new batch with document states initialized as PENDIENTE."""
        batch_id = str(uuid.uuid4())
        documents: Dict[str, DocumentExtractionState] = {}
        for fid, fname in zip(file_ids, filenames):
            documents[fid] = DocumentExtractionState(id=fid, filename=fname)
        self._batches[batch_id] = BatchState(
            batch_id=batch_id,
            documents=documents,
            created_at=datetime.now().isoformat(),
        )
        return batch_id

    def update_document_status(
        self,
        batch_id: str,
        doc_id: str,
        status: ExtractionStatus,
        **kwargs,
    ) -> None:
        """Update extraction status and optional fields for a document."""
        batch = self._batches.get(batch_id)
        if batch is None:
            return
        doc = batch.documents.get(doc_id)
        if doc is None:
            return
        doc.status = status
        for key, value in kwargs.items():
            if hasattr(doc, key):
                setattr(doc, key, value)

    def get_batch_status(self, batch_id: str) -> Optional[BatchState]:
        """Return the batch state or None if not found."""
        return self._batches.get(batch_id)

    def add_documents_to_batch(
        self, batch_id: str, file_ids: List[str], filenames: List[str]
    ) -> None:
        """Add new documents to an existing batch (e.g. reference doc uploads)."""
        batch = self._batches.get(batch_id)
        if batch is None:
            return
        for fid, fname in zip(file_ids, filenames):
            batch.documents[fid] = DocumentExtractionState(id=fid, filename=fname)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_batch_manager: Optional[BatchManager] = None


def get_batch_manager() -> BatchManager:
    """Return the singleton BatchManager instance."""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchManager()
    return _batch_manager
