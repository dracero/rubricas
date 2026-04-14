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

# Maximum characters sent to the LLM to stay within token limits
_MAX_TEXT_CHARS = 25_000


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
        """Extract ontology and references from *document_text*."""

        if not document_text or not document_text.strip():
            return ExtractionResult(
                success=False,
                error_message="Texto vacío o ilegible",
            )

        try:
            ontologia = await self._extract_ontology(document_text)
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
    # Ontology extraction
    # ------------------------------------------------------------------

    async def _extract_ontology(self, text: str) -> Ontologia:
        """Call OpenAI via LiteLLM to extract entities and relations."""

        truncated = text[:_MAX_TEXT_CHARS]

        system_prompt = (
            "Eres un experto en análisis de documentos normativos. "
            "Tu tarea es extraer una ontología estructurada del texto proporcionado.\n\n"
            "Debes identificar:\n"
            "- **Entidades**: elementos clave del documento (normas, requisitos, actores, "
            "procesos, documentos, secciones, etc.). Cada entidad tiene: nombre, tipo, "
            "contexto (breve descripción de su rol en el documento) y propiedades "
            "(diccionario con atributos adicionales relevantes).\n"
            "- **Relaciones**: vínculos entre entidades. Cada relación tiene: origen "
            "(nombre de entidad), destino (nombre de entidad), tipo (verbo o frase que "
            "describe la relación) y propiedades (diccionario con atributos adicionales).\n\n"
            "Responde ÚNICAMENTE con un objeto JSON válido con esta estructura:\n"
            "{\n"
            '  "entidades": [\n'
            '    {"nombre": "...", "tipo": "...", "contexto": "...", "propiedades": {}}\n'
            "  ],\n"
            '  "relaciones": [\n'
            '    {"origen": "...", "destino": "...", "tipo": "...", "propiedades": {}}\n'
            "  ]\n"
            "}\n\n"
            "Extrae al menos 5 entidades y 3 relaciones por entidad cuando el texto lo permita. "
            "No incluyas explicaciones fuera del JSON."
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated},
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content or ""
        data = parsear_json_con_fallback(raw)

        entidades: List[Entidad] = []
        for e in data.get("entidades", []):
            entidades.append(
                Entidad(
                    nombre=e.get("nombre", ""),
                    tipo=e.get("tipo", ""),
                    contexto=e.get("contexto", ""),
                    propiedades=e.get("propiedades", {}),
                    fecha_creacion=datetime.now().isoformat(),
                )
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
            entidades=entidades,
            relaciones=relaciones,
            metadata={"extracted_at": datetime.now().isoformat()},
        )

    # ------------------------------------------------------------------
    # Reference detection
    # ------------------------------------------------------------------

    async def _detect_references(self, text: str) -> List[DocumentReference]:
        """Call OpenAI via LiteLLM to detect external document references."""

        truncated = text[:_MAX_TEXT_CHARS]

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
            "- \"text\": el texto de la referencia tal como aparece en el documento\n"
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
                    {"role": "user", "content": truncated},
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
