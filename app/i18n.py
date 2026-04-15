"""
Per-session internationalization helpers for RubricAI backend.

Provides language extraction from requests, translated backend messages,
and language name mappings for agent directives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

SUPPORTED_LANGUAGES = ['es', 'gl', 'en', 'pt']
DEFAULT_LANGUAGE = 'es'

LANGUAGE_NAMES = {
    'es': 'español',
    'gl': 'galego',
    'en': 'English',
    'pt': 'português',
}

BACKEND_MESSAGES = {
    'es': {
        'empty_message': 'Mensaje vacío',
        'doc_not_found': 'Documento no encontrado',
        'rubric_not_found': 'Rúbrica no encontrada',
        'pdf_only': 'Solo se aceptan archivos PDF',
        'generation_error': 'Error generando rúbrica',
        'evaluation_error': 'Error evaluando',
        'no_text_extracted': 'No se pudo extraer texto del PDF',
        'topic_mismatch': 'El documento no tiene suficiente relación temática con la rúbrica.',
        'institution_mismatch': 'El documento no pertenece a la institución configurada.',
        'file_not_found': 'Archivo no encontrado',
        'batch_not_found': 'Lote no encontrado',
        'incompatible_doc': 'El documento no es compatible con esta rúbrica.',
    },
    'gl': {
        'empty_message': 'Mensaxe baleira',
        'doc_not_found': 'Documento non atopado',
        'rubric_not_found': 'Rúbrica non atopada',
        'pdf_only': 'Só se aceptan ficheiros PDF',
        'generation_error': 'Erro xerando rúbrica',
        'evaluation_error': 'Erro avaliando',
        'no_text_extracted': 'Non se puido extraer texto do PDF',
        'topic_mismatch': 'O documento non ten suficiente relación temática coa rúbrica.',
        'institution_mismatch': 'O documento non pertence á institución configurada.',
        'file_not_found': 'Ficheiro non atopado',
        'batch_not_found': 'Lote non atopado',
        'incompatible_doc': 'O documento non é compatible con esta rúbrica.',
    },
    'en': {
        'empty_message': 'Empty message',
        'doc_not_found': 'Document not found',
        'rubric_not_found': 'Rubric not found',
        'pdf_only': 'Only PDF files are accepted',
        'generation_error': 'Error generating rubric',
        'evaluation_error': 'Evaluation error',
        'no_text_extracted': 'Could not extract text from PDF',
        'topic_mismatch': 'The document does not have enough thematic relation with the rubric.',
        'institution_mismatch': 'The document does not belong to the configured institution.',
        'file_not_found': 'File not found',
        'batch_not_found': 'Batch not found',
        'incompatible_doc': 'The document is not compatible with this rubric.',
    },
    'pt': {
        'empty_message': 'Mensagem vazia',
        'doc_not_found': 'Documento não encontrado',
        'rubric_not_found': 'Rubrica não encontrada',
        'pdf_only': 'Apenas ficheiros PDF são aceites',
        'generation_error': 'Erro ao gerar rubrica',
        'evaluation_error': 'Erro na avaliação',
        'no_text_extracted': 'Não foi possível extrair texto do PDF',
        'topic_mismatch': 'O documento não tem relação temática suficiente com a rubrica.',
        'institution_mismatch': 'O documento não pertence à instituição configurada.',
        'file_not_found': 'Ficheiro não encontrado',
        'batch_not_found': 'Lote não encontrado',
        'incompatible_doc': 'O documento não é compatível com esta rubrica.',
    },
}


def get_request_language(request: "Request") -> str:
    """Extract and validate language from Accept-Language header.

    Parses the first language code from the header value, ignoring quality
    parameters and additional languages.  Returns DEFAULT_LANGUAGE when the
    header is missing or contains an unsupported code.
    """
    lang = request.headers.get('accept-language', DEFAULT_LANGUAGE)
    # Take first language code, ignore quality values (e.g. "en-US;q=0.9,gl")
    lang = lang.split(',')[0].split(';')[0].strip().lower()[:2]
    if lang not in SUPPORTED_LANGUAGES:
        return DEFAULT_LANGUAGE
    return lang


def get_message(key: str, lang: str) -> str:
    """Get a translated backend message.

    Falls back to the DEFAULT_LANGUAGE translation, then to the raw key
    if the key is not found in any language.
    """
    messages = BACKEND_MESSAGES.get(lang, BACKEND_MESSAGES[DEFAULT_LANGUAGE])
    return messages.get(key, BACKEND_MESSAGES[DEFAULT_LANGUAGE].get(key, key))
