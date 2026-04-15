import React, { useState, useEffect, useRef } from 'react';
import { Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

const statusIcon = (status) => {
    switch (status) {
        case 'en_proceso':
            return <Loader2 className="w-4 h-4 text-blue-500 animate-spin shrink-0" />;
        case 'completado':
            return <CheckCircle className="w-4 h-4 text-green-500 shrink-0" />;
        case 'error':
            return <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />;
        default:
            return <Clock className="w-4 h-4 text-gray-400 shrink-0" />;
    }
};

const ExtractionProgress = ({ batchId, onComplete }) => {
    const { lang, t } = useLanguage();
    const [statusData, setStatusData] = useState(null);
    const [pollError, setPollError] = useState('');
    const intervalRef = useRef(null);
    const completedRef = useRef(false);

    const statusLabel = (status) => {
        switch (status) {
            case 'en_proceso': return t('extraction.status.processing');
            case 'completado': return t('extraction.status.completed');
            case 'error': return t('extraction.status.error');
            default: return t('extraction.status.pending');
        }
    };

    useEffect(() => {
        if (!batchId) return;

        const poll = async () => {
            try {
                const res = await fetch(`/api/upload/status/${batchId}`, {
                    headers: { 'Accept-Language': lang },
                });
                if (!res.ok) throw new Error(t('extraction.poll.error'));
                const data = await res.json();
                setStatusData(data);
                setPollError('');

                const allDone = data.documents?.every(
                    d => d.status === 'completado' || d.status === 'error'
                );
                if (allDone && !completedRef.current) {
                    completedRef.current = true;
                    clearInterval(intervalRef.current);
                    if (onComplete) onComplete(data);
                }
            } catch {
                setPollError(t('extraction.poll.error'));
            }
        };

        poll();
        intervalRef.current = setInterval(poll, 3000);

        return () => clearInterval(intervalRef.current);
    }, [batchId, onComplete]);

    if (!statusData) {
        return (
            <div className="flex items-center gap-2 text-sm text-gray-500 py-4">
                <Loader2 className="w-4 h-4 animate-spin" />
                {t('extraction.loading')}
            </div>
        );
    }

    const { documents, summary } = statusData;
    const allDone = documents?.every(d => d.status === 'completado' || d.status === 'error');

    return (
        <div className="space-y-3">
            <p className="text-sm font-medium text-gray-700">
                {t('extraction.title')}
            </p>

            <ul className="space-y-2">
                {documents?.map(doc => (
                    <li key={doc.id} className="flex items-start gap-2 bg-gray-50 px-3 py-2 rounded border border-gray-200 text-sm">
                        {statusIcon(doc.status)}
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <span className="truncate font-medium">{doc.filename}</span>
                                <span className="text-xs text-gray-400">{statusLabel(doc.status)}</span>
                            </div>
                            {doc.status === 'completado' && (
                                <p className="text-xs text-gray-500 mt-0.5">
                                    {doc.entities_count} {t('extraction.entities')} · {doc.relations_count} {t('extraction.relations')}
                                </p>
                            )}
                            {doc.status === 'error' && doc.error_message && (
                                <p className="text-xs text-red-500 mt-0.5">{doc.error_message}</p>
                            )}
                        </div>
                    </li>
                ))}
            </ul>

            {pollError && (
                <p className="text-xs text-yellow-600">{pollError}</p>
            )}

            {allDone && summary && (
                <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm">
                    <p className="font-medium text-blue-800">{t('extraction.summary.title')}</p>
                    <p className="text-blue-700 text-xs mt-1">
                        {summary.total} documento{summary.total > 1 ? 's' : ''} procesado{summary.total > 1 ? 's' : ''} —{' '}
                        {summary.completado} exitoso{summary.completado > 1 ? 's' : ''}{summary.error > 0 ? `, ${summary.error} con error` : ''}
                    </p>
                </div>
            )}
        </div>
    );
};

export default ExtractionProgress;
