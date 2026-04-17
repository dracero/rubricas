import React, { useState, useEffect } from 'react';
import { PenTool, Loader2, CheckCircle, AlertCircle, BookOpen } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

const WritingAssistant = ({ onRubricSelected }) => {
    const { lang, t } = useLanguage();
    const [repoRubrics, setRepoRubrics] = useState([]);
    const [selectedRubric, setSelectedRubric] = useState('');
    const [rubricContent, setRubricContent] = useState('');
    const [status, setStatus] = useState('idle');
    const [error, setError] = useState('');
    const [loadingRubrics, setLoadingRubrics] = useState(true);

    // Load rubrics from repo on mount
    useEffect(() => {
        const fetchRubrics = async () => {
            try {
                const res = await fetch('/api/rubrics/files', {
                    headers: { 'Accept-Language': lang },
                });
                if (res.ok) {
                    const data = await res.json();
                    setRepoRubrics(data.files || []);
                }
            } catch { /* ignore */ }
            finally { setLoadingRubrics(false); }
        };
        fetchRubrics();
    }, []);

    const handleLoadRubric = async () => {
        if (!selectedRubric) return;
        setStatus('loading');
        setError('');

        try {
            // Download the rubric content to show a preview
            const res = await fetch(`/api/rubrics/files/${encodeURIComponent(selectedRubric)}`, {
                headers: { 'Accept-Language': lang },
            });
            if (!res.ok) throw new Error('Error al cargar la rúbrica');

            // Get rubric metadata (topics) from the list
            const rubricMeta = repoRubrics.find(r => r.filename === selectedRubric);
            const topics = rubricMeta?.topics || [];

            setRubricContent(`Rúbrica: ${selectedRubric}\nTemas: ${topics.join(', ') || 'N/A'}`);
            setStatus('loaded');

            // Notify parent with rubric info so it can send context to the chat
            if (onRubricSelected) {
                onRubricSelected({
                    filename: selectedRubric,
                    topics,
                });
            }
        } catch (err) {
            console.error(err);
            setError(err.message);
            setStatus('error');
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4 max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <PenTool className="w-5 h-5 text-teal-600" />
                {t('writer.title') || 'Asistente de Redacción'}
            </h3>

            <p className="text-sm text-gray-600 mb-4">
                {t('writer.description') || 'Selecciona una rúbrica de referencia. El asistente usará sus criterios para guiar la redacción y asegurar el cumplimiento.'}
            </p>

            {status === 'loaded' ? (
                <div className="space-y-3">
                    <div className="flex items-center gap-2 text-green-700">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">
                            {t('writer.rubric.loaded') || 'Rúbrica cargada como referencia'}
                        </span>
                    </div>
                    <div className="bg-teal-50 p-3 rounded-lg border border-teal-200 text-sm text-teal-800 flex items-start gap-2">
                        <BookOpen className="w-4 h-4 mt-0.5 shrink-0" />
                        <div>
                            <p className="font-medium">{selectedRubric}</p>
                            {repoRubrics.find(r => r.filename === selectedRubric)?.topics?.length > 0 && (
                                <p className="text-teal-600 mt-1">
                                    {t('writer.topics') || 'Temas'}: {repoRubrics.find(r => r.filename === selectedRubric).topics.join(', ')}
                                </p>
                            )}
                        </div>
                    </div>
                    <p className="text-sm text-gray-500">
                        {t('writer.ready') || 'Ahora puedes escribir en el chat y el asistente validará contra esta rúbrica.'}
                    </p>
                    <button
                        onClick={() => { setStatus('idle'); setSelectedRubric(''); setRubricContent(''); }}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700">
                        {t('writer.change.rubric') || 'Cambiar rúbrica'}
                    </button>
                </div>
            ) : (
                <div className="space-y-4">
                    {/* Rubric dropdown from repo */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            {t('writer.rubric.label') || 'Rúbrica de referencia'}
                        </label>
                        {loadingRubrics ? (
                            <div className="flex items-center gap-2 text-sm text-gray-500 py-2">
                                <Loader2 className="w-4 h-4 animate-spin" /> {t('evaluator.loading.rubrics') || 'Cargando rúbricas...'}
                            </div>
                        ) : repoRubrics.length === 0 ? (
                            <p className="text-sm text-gray-500 py-2">
                                {t('writer.no.rubrics') || 'No hay rúbricas en el repositorio. Genera una primero.'}
                            </p>
                        ) : (
                            <select
                                value={selectedRubric}
                                onChange={(e) => setSelectedRubric(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-teal-500 focus:outline-none text-sm"
                            >
                                <option value="">{t('writer.rubric.select') || 'Selecciona una rúbrica...'}</option>
                                {repoRubrics.map((r) => (
                                    <option key={r.filename} value={r.filename}>
                                        {r.topics?.length > 0 ? `[${r.topics.slice(0, 3).join(', ')}] ` : ''}{r.filename}
                                    </option>
                                ))}
                            </select>
                        )}
                    </div>

                    {error && (
                        <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded">
                            <AlertCircle className="w-4 h-4" /> {error}
                        </div>
                    )}

                    <button
                        onClick={handleLoadRubric}
                        disabled={!selectedRubric || status === 'loading'}
                        className="w-full bg-teal-600 text-white py-2 px-4 rounded hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
                    >
                        {status === 'loading' ? (
                            <><Loader2 className="w-4 h-4 animate-spin" /> {t('writer.button.loading') || 'Cargando...'}</>
                        ) : (t('writer.button.load') || 'Cargar rúbrica de referencia')}
                    </button>
                </div>
            )}
        </div>
    );
};

export default WritingAssistant;
