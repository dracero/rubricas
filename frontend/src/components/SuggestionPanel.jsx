import React, { useState } from 'react';
import { Sparkles, Eye, FileText, Calendar, BarChart3, Loader2, X, Download } from 'lucide-react';
import MarkdownTable from './MarkdownTable';

const levelLabel = (level) => {
    switch (level) {
        case 'inicial': return 'Operacional';
        case 'avanzado': return 'Técnico';
        case 'critico': return 'Alta Criticidad';
        default: return level;
    }
};

const levelColor = (level) => {
    switch (level) {
        case 'inicial': return 'bg-green-100 text-green-700';
        case 'avanzado': return 'bg-blue-100 text-blue-700';
        case 'critico': return 'bg-red-100 text-red-700';
        default: return 'bg-gray-100 text-gray-700';
    }
};

const formatDate = (isoStr) => {
    if (!isoStr) return '—';
    try {
        return new Date(isoStr).toLocaleDateString('es-ES', {
            day: 'numeric', month: 'short', year: 'numeric',
        });
    } catch {
        return isoStr;
    }
};

const formatScore = (score) => {
    if (score == null) return '—';
    return `${Math.round(score * 100)}%`;
};

const SuggestionPanel = ({ suggestions, onSelectBase, onGenerateNew, onViewFull }) => {
    const [expandedId, setExpandedId] = useState(null);
    const [fullRubric, setFullRubric] = useState(null);
    const [loadingFull, setLoadingFull] = useState(null);

    const handleViewFull = async (rubricId) => {
        if (expandedId === rubricId) {
            setExpandedId(null);
            setFullRubric(null);
            return;
        }

        setLoadingFull(rubricId);
        try {
            const res = await fetch(`/api/rubrics/${rubricId}`);
            if (!res.ok) throw new Error('Error al obtener rúbrica');
            const data = await res.json();
            setFullRubric(data);
            setExpandedId(rubricId);
        } catch (err) {
            console.error(err);
        } finally {
            setLoadingFull(null);
        }
    };

    if (!suggestions || suggestions.length === 0) return null;

    return (
        <div className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm text-blue-700 flex items-center gap-2">
                <Sparkles className="w-4 h-4 shrink-0" />
                Se encontraron <strong>{suggestions.length}</strong> rúbrica{suggestions.length > 1 ? 's' : ''} similar{suggestions.length > 1 ? 'es' : ''} en el repositorio
            </div>

            <div className="space-y-3">
                {suggestions.map((s) => (
                    <div key={s.rubric_id} className="bg-white border border-gray-200 rounded-lg p-4 space-y-3">
                        {/* Header: score + level + date */}
                        <div className="flex items-center gap-2 flex-wrap">
                            <span className="inline-flex items-center gap-1 text-sm font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded">
                                <BarChart3 className="w-3.5 h-3.5" />
                                {formatScore(s.score)}
                            </span>
                            <span className={`text-xs px-2 py-0.5 rounded font-medium ${levelColor(s.level)}`}>
                                {levelLabel(s.level)}
                            </span>
                            <span className="text-xs text-gray-400 flex items-center gap-1">
                                <Calendar className="w-3 h-3" />
                                {formatDate(s.created_at)}
                            </span>
                        </div>

                        {/* Source documents */}
                        {s.source_filenames?.length > 0 && (
                            <div className="flex items-center gap-1 flex-wrap text-xs text-gray-500">
                                <FileText className="w-3 h-3 shrink-0" />
                                {s.source_filenames.join(', ')}
                            </div>
                        )}

                        {/* Text preview */}
                        <p className="text-sm text-gray-600 leading-relaxed line-clamp-3">
                            {s.summary || '(Sin resumen)'}
                        </p>

                        {/* Expanded full rubric */}
                        {expandedId === s.rubric_id && fullRubric && (
                            <div className="bg-gray-50 border border-gray-200 rounded p-3 max-h-64 overflow-y-auto relative">
                                <button
                                    onClick={() => { setExpandedId(null); setFullRubric(null); }}
                                    className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                                <MarkdownTable content={fullRubric.rubric_text || 'Sin contenido'} />
                                {fullRubric.download_url && (
                                    <a
                                        href={fullRubric.download_url}
                                        download
                                        className="inline-flex items-center gap-1 mt-2 text-xs text-blue-600 hover:text-blue-800"
                                    >
                                        <Download className="w-3 h-3" />
                                        Descargar DOCX
                                    </a>
                                )}
                            </div>
                        )}

                        {/* Action buttons */}
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => onSelectBase(s.rubric_id)}
                                className="text-xs bg-blue-600 text-white px-3 py-1.5 rounded hover:bg-blue-700 transition"
                            >
                                Usar como base
                            </button>
                            <button
                                onClick={() => handleViewFull(s.rubric_id)}
                                disabled={loadingFull === s.rubric_id}
                                className="text-xs border border-gray-300 text-gray-600 px-3 py-1.5 rounded hover:bg-gray-50 transition disabled:opacity-50 flex items-center gap-1"
                            >
                                {loadingFull === s.rubric_id ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                    <Eye className="w-3 h-3" />
                                )}
                                {expandedId === s.rubric_id ? 'Ocultar' : 'Ver completa'}
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {/* Global generate new button */}
            <button
                onClick={onGenerateNew}
                className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded hover:bg-gray-200 transition text-sm flex items-center justify-center gap-2"
            >
                <Sparkles className="w-4 h-4" />
                Generar nueva (ignorar sugerencias)
            </button>
        </div>
    );
};

export default SuggestionPanel;
