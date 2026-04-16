"""
Ontology extraction service using OpenAI via LiteLLM.

Extracts entities, relations, and external document references from
normative PDF text without going through the full ADK agent pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import litellm

from app.domain import Entidad, Relacion, Ontologia, parsear_json_con_fallback
from app.models import DocumentReference

logger = logging.getLogger(__name__)

# Maximum characters per chunk sent to the LLM to stay within token limits
_CHUNK_SIZE = 20_000
# Overlap between chunks to avoid losing context at boundaries
_CHUNK_OVERLAP = 2_000


@dataclass
class ExtractionResult:
    """Result of ontology extraction for a single document."""

    ontologia: Optional[Ontologia] = None
    references: List[DocumentReference] = field(default_factory=list)
    success: bool = False
    error_message: str = ""
    entities_count: int = 0
    relations_count: int = 0


class OntologyExtractor:
    """Extracts ontology from document text using OpenAI via LiteLLM."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self,
        document_text: str,
        document_id: str,
        source_filename: str,
    ) -> ExtractionResult:
        """Extract ontology and references from *document_text*.

        If the document exceeds _CHUNK_SIZE characters it is processed in
        overlapping chunks and the results are merged, so no content is
        silently dropped.
        """

        if not document_text or not document_text.strip():
            return ExtractionResult(
                success=False,
                error_message="Texto vacío o ilegible",
            )

        try:
            if len(document_text) > _CHUNK_SIZE:
                logger.warning(
                    "⚠️  Documento '%s' (%s) supera %d chars — procesando en chunks",
                    source_filename,
                    document_id,
                    _CHUNK_SIZE,
                )
                return await self._extract_chunked(document_text, document_id, source_filename)

            ontologia = await self._extract_ontology(document_text, source_filename)
            references = await self._detect_references(document_text)

            return ExtractionResult(
                ontologia=ontologia,
                references=references,
                success=True,
                entities_count=len(ontologia.entidades),
                relations_count=len(ontologia.relaciones),
            )
        except Exception as exc:
            logger.exception(
                "Error extracting ontology from %s (%s): %s",
                source_filename,
                document_id,
                exc,
            )
            return ExtractionResult(
                success=False,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Chunked extraction for long documents
    # ------------------------------------------------------------------

    async def _extract_chunked(
        self,
        text: str,
        document_id: str,
        source_filename: str,
    ) -> ExtractionResult:
        """Process long documents in overlapping chunks and merge results.

        Ensures every part of the document is processed — nothing is silently
        truncated.
        """
        chunks = self._build_chunks(text)
        total = len(chunks)
        logger.info(
            "📄 '%s' dividido en %d chunks (chunk=%d, overlap=%d)",
            source_filename,
            total,
            _CHUNK_SIZE,
            _CHUNK_OVERLAP,
        )

        all_entidades: List[Entidad] = []
        all_relaciones: List[Relacion] = []
        all_references: List[DocumentReference] = []

        for i, chunk in enumerate(chunks):
            logger.info("  🔍 Procesando chunk %d/%d de '%s'", i + 1, total, source_filename)
            try:
                ontologia = await self._extract_ontology(chunk, source_filename, chunk_index=i + 1)
                refs = await self._detect_references(chunk)
                all_entidades.extend(ontologia.entidades)
                all_relaciones.extend(ontologia.relaciones)
                all_references.extend(refs)
            except Exception as exc:
                logger.error("  ❌ Error en chunk %d/%d de '%s': %s", i + 1, total, source_filename, exc)
                # Continue with remaining chunks — partial result is better than none

        # Deduplicate entities by name (keep first occurrence)
        seen: set = set()
        unique_entidades: List[Entidad] = []
        for e in all_entidades:
            key = e.nombre.lower().strip()
            if key not in seen:
                seen.add(key)
                unique_entidades.append(e)

        # Deduplicate references by text
        seen_refs: set = set()
        unique_references: List[DocumentReference] = []
        for r in all_references:
            key = r.text.lower().strip()
            if key not in seen_refs:
                seen_refs.add(key)
                unique_references.append(r)

        logger.info(
            "✅ Merge completado para '%s': %d entidades únicas, %d relaciones, %d referencias",
            source_filename,
            len(unique_entidades),
            len(all_relaciones),
            len(unique_references),
        )

        return ExtractionResult(
            ontologia=Ontologia(
                entidades=unique_entidades,
                relaciones=all_relaciones,
                metadata={
                    "extracted_at": datetime.now().isoformat(),
                    "chunks_processed": total,
                    "source_filename": source_filename,
                },
            ),
            references=unique_references,
            success=True,
            entities_count=len(unique_entidades),
            relations_count=len(all_relaciones),
        )

    @staticmethod
    def _build_chunks(text: str) -> List[str]:
        """Split *text* into overlapping chunks of _CHUNK_SIZE chars."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + _CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += _CHUNK_SIZE - _CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------
    # Ontology extraction (single chunk)
    # ------------------------------------------------------------------

    async def _extract_ontology(self, text: str, source_filename: str = "", chunk_index: Optional[int] = None) -> Ontologia:
        """Call OpenAI via LiteLLM to extract entities and relations."""

        chunk_note = f" (fragmento {chunk_index})" if chunk_index else ""

        system_prompt = (
            "Eres un experto en análisis de documentos normativos. "
            "Tu tarea es extraer una ontología estructurada del texto proporcionado.\n\n"
            "REGLA CRÍTICA — ANTI-ALUCINACIÓN:\n"
            "Solo extrae entidades y relaciones que estén EXPLÍCITAMENTE mencionadas "
            "en el texto proporcionado. NO inferas, NO completes con conocimiento externo, "
            "NO inventes entidades para llegar a un mínimo numérico. "
            "Si el texto tiene pocas entidades claras, extrae solo las que realmente existan.\n\n"
            "Debes identificar:\n"
            "- **Entidades**: elementos clave del documento (normas, requisitos, actores, "
            "procesos, documentos, secciones, etc.). Cada entidad tiene: nombre, tipo, "
            "contexto (cita textual o paráfrasis directa del documento — NO descripción genérica) "
            "y propiedades (diccionario con atributos adicionales relevantes).\n"
            "- **Relaciones**: vínculos entre entidades MENCIONADOS en el texto. "
            "Cada relación tiene: origen (nombre de entidad), destino (nombre de entidad), "
            "tipo (verbo o frase que describe la relación) y propiedades.\n\n"
            "Responde ÚNICAMENTE con un objeto JSON válido con esta estructura:\n"
            "{\n"
            '  "entidades": [\n'
            '    {"nombre": "...", "tipo": "...", "contexto": "...", "propiedades": {}}\n'
            "  ],\n"
            '  "relaciones": [\n'
            '    {"origen": "...", "destino": "...", "tipo": "...", "propiedades": {}}\n'
            "  ]\n"
            "}\n\n"
            "No incluyas explicaciones fuera del JSON."
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Documento: {source_filename}{chunk_note}\n\n"
                        f"{text}"
                    ),
                },
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content or ""
        data = parsear_json_con_fallback(raw)

        entidades: List[Entidad] = []
        for e in data.get("entidades", []):
            nombre = e.get("nombre", "").strip()
            if not nombre:
                continue
            entidades.append(
                Entidad(
                    nombre=nombre,
                    tipo=e.get("tipo", ""),
                    contexto=e.get("contexto", ""),
                    propiedades=e.get("propiedades", {}),
                    fecha_creacion=datetime.now().isoformat(),
                )
            )

        # Post-extraction validation: discard entities whose key words cannot be
        # found anywhere in the source text (flexible grounding check).
        # Instead of requiring the exact entity name, we check that at least
        # half of the significant words (3+ chars) appear in the text.
        text_lower = text.lower()
        validated_entidades: List[Entidad] = []
        for e in entidades:
            name_words = [w for w in e.nombre.lower().split("_") if len(w) >= 3]
            if not name_words:
                name_words = [e.nombre.lower().strip()]
            
            # Check how many key words appear in the text
            found_count = sum(1 for w in name_words if w in text_lower)
            threshold = max(1, len(name_words) // 2)  # at least half the words
            
            if found_count >= threshold:
                validated_entidades.append(e)
            else:
                logger.warning(
                    "⚠️  Entidad '%s' — solo %d/%d palabras encontradas en el texto — descartada",
                    e.nombre, found_count, len(name_words),
                )

        relaciones: List[Relacion] = []
        for r in data.get("relaciones", []):
            relaciones.append(
                Relacion(
                    origen=r.get("origen", ""),
                    destino=r.get("destino", ""),
                    tipo=r.get("tipo", ""),
                    propiedades=r.get("propiedades", {}),
                )
            )

        return Ontologia(
            entidades=validated_entidades,
            relaciones=relaciones,
            metadata={"extracted_at": datetime.now().isoformat()},
        )

    # ------------------------------------------------------------------
    # Reference detection
    # ------------------------------------------------------------------

    async def _detect_references(self, text: str) -> List[DocumentReference]:
        """Call OpenAI via LiteLLM to detect external document references."""

        system_prompt = (
            "Eres un experto en análisis de documentos normativos. "
            "Tu tarea es identificar todas las referencias a documentos externos "
            "mencionados en el texto proporcionado.\n\n"
            "Busca:\n"
            "- Normas ISO, IEC, UNE, ASTM u otras normas técnicas\n"
            "- Leyes, decretos, resoluciones y regulaciones\n"
            "- URLs o hipervínculos a documentos externos\n"
            "- Códigos de referencia documental internos o externos\n\n"
            "Responde ÚNICAMENTE con un arreglo JSON válido. Cada elemento debe tener:\n"
            "- \"type\": uno de \"normativa\", \"url\" o \"codigo_referencia\"\n"
            "- \"text\": el texto de la referencia TAL COMO APARECE en el documento\n"
            "- \"url\": la URL si aplica, o null\n\n"
            "Ejemplo:\n"
            "[\n"
            '  {"type": "normativa", "text": "ISO 9001:2015", "url": null},\n'
            '  {"type": "url", "text": "https://example.com/doc", "url": "https://example.com/doc"}\n'
            "]\n\n"
            "Si no encuentras referencias, responde con un arreglo vacío: []\n"
            "No incluyas explicaciones fuera del JSON."
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
            )

            raw = response.choices[0].message.content or ""

            # Try to parse as a JSON array directly
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                # Strip markdown code fences and retry
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
                cleaned = cleaned.strip()
                try:
                    items = json.loads(cleaned)
                except json.JSONDecodeError:
                    logger.warning("Could not parse references JSON: %s", raw[:200])
                    return []

            if not isinstance(items, list):
                return []

            references: List[DocumentReference] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                references.append(
                    DocumentReference(
                        type=item.get("type", "normativa"),
                        text=item.get("text", ""),
                        url=item.get("url"),
                    )
                )
            return references

        except Exception as exc:
            logger.warning("Error detecting references: %s", exc)
            return []
