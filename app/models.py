"""
Pydantic response models for the multi-document ontology extraction API
and the rubric repository.
"""

from typing import List, Optional

from pydantic import BaseModel


class DocumentReference(BaseModel):
    type: str
    text: str
    url: Optional[str] = None


class FileInfo(BaseModel):
    id: str
    filename: str


class RejectedFile(BaseModel):
    filename: str
    reason: str


class BatchUploadResponse(BaseModel):
    batch_id: str
    accepted: List[FileInfo]
    rejected: List[RejectedFile]


class DocumentStatus(BaseModel):
    id: str
    filename: str
    status: str
    entities_count: int = 0
    relations_count: int = 0
    error_message: str = ""
    references: List[DocumentReference] = []


class BatchSummary(BaseModel):
    total: int
    completado: int
    en_proceso: int
    error: int
    pendiente: int


class BatchStatusResponse(BaseModel):
    batch_id: str
    documents: List[DocumentStatus]
    summary: BatchSummary


# ============================================================================
# Rubric Repository models
# ============================================================================


class RubricSummary(BaseModel):
    """Summary of a rubric for listings and suggestions."""
    rubric_id: str
    summary: str
    level: str
    source_filenames: List[str]
    created_at: str
    score: Optional[float] = None


class RubricDetail(BaseModel):
    """Full rubric from the repository."""
    rubric_id: str
    rubric_text: str
    level: str
    source_filenames: List[str]
    source_document_ids: List[str]
    created_at: str
    download_url: Optional[str] = None


class RubricListResponse(BaseModel):
    """Paginated rubric list response."""
    rubrics: List[RubricSummary]
    total: int
    limit: int
    offset: int


class GenerateResponse(BaseModel):
    """Extended response for the /api/generate endpoint."""
    result: str
    download_url: str
    similar_rubrics: List[RubricSummary] = []
