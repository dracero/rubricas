import React, { useState, useEffect } from 'react';
import { Upload, FileSearch, Loader2, CheckCircle, AlertCircle, Download } from 'lucide-react';
import MarkdownTable from './MarkdownTable';
import { useLanguage } from '../contexts/LanguageContext';

const RubricEvaluator = ({ onComplete }) => {
    const { lang, t } = useLanguage();
    const [repoRubrics, setRepoRubrics] = useState([]);
    const [selectedRubric, setSelectedRubric] = useState('');
    const [docFile, setDocFile] = useState(null);
    const [docId, setDocId] = useState('');
    const [status, setStatus] = useState('idle');
    const [result, setResult] = useState(null);
    const [correctedResult, setCorrectedResult] = useState(null);
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

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedRubric || !docFile) return;

        setStatus('processing');
        setError('');

        try {
            // 1. Upload Doc
            const formDoc = new FormData();
            formDoc.append('file', docFile);
            const docRes = await fetch('/api/evaluate/upload_doc', {
                method: 'POST',
                headers: { 'Accept-Language': lang },
                body: formDoc
            });
            if (!docRes.ok) throw new Error(t('evaluator.error.upload'));
            const docData = await docRes.json();
            setDocId(docData.id);

            // 2. Evaluate using repo rubric filename
            const evalRes = await fetch('/api/evaluate/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept-Language': lang,
                },
                body: JSON.stringify({
                    rubric_filename: selectedRubric,
                    doc_id: docData.id
                })
            });

            if (!evalRes.ok) throw new Error(t('evaluator.error.evaluate'));
            const evalData = await evalRes.json();

            // Check for topic mismatch warning
            if (evalData.topic_mismatch && evalData.can_force) {
                setResult(evalData);
                setStatus('mismatch');
                return;
            }

            setResult(evalData);
            setStatus('success');
            if (onComplete) onComplete(evalData);

        } catch (err) {
            console.error(err);
            setError(err.message);
            setStatus('error');
        }
    };

    const handleForceEvaluate = async () => {
        setStatus('processing');
        setError('');
        try {
            const evalRes = await fetch('/api/evaluate/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept-Language': lang,
                },
                body: JSON.stringify({
                    rubric_filename: selectedRubric,
                    doc_id: docId,
                    force: true,
                })
            });
            if (!evalRes.ok) throw new Error(t('evaluator.error.evaluate'));
            const evalData = await evalRes.json();
            setResult(evalData);
            setStatus('success');
            if (onComplete) onComplete(evalData);
        } catch (err) {
            setError(err.message);
            setStatus('error');
        }
    };

    const handleAutoCorrect = async () => {
        setStatus('correcting');
        setError('');
        try {
            const res = await fetch('/api/evaluate/autocorrect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept-Language': lang,
                },
                body: JSON.stringify({
                    rubric_filename: selectedRubric,
                    doc_id: docId,
                    evaluation_result: result?.result || '',
                })
            });
            if (!res.ok) throw new Error('Error corrigiendo documento');
            const data = await res.json();
            setCorrectedResult(data);
            setStatus('corrected');
        } catch (err) {
            setError(err.message);
            setStatus('success');
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4 max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FileSearch className="w-5 h-5 text-purple-600" />
                {t('evaluator.title')}
            </h3>

            {status === 'mismatch' ? (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-amber-700 bg-amber-50 p-4 rounded-lg border border-amber-200">
                        <AlertCircle className="w-5 h-5 shrink-0" />
                        <div>
                            <p className="font-medium">{t('evaluator.mismatch.title') || '⚠️ Posible incompatibilidad detectada'}</p>
                            <p className="text-sm mt-1">{result?.mismatch_reason || 'La rúbrica y el documento podrían no ser del mismo tema o institución.'}</p>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={handleForceEvaluate}
                            className="flex-1 bg-amber-600 text-white py-2 px-4 rounded hover:bg-amber-700 transition text-sm"
                        >
                            {t('evaluator.force') || 'Evaluar de todas formas'}
                        </button>
                        <button
                            onClick={() => { setStatus('idle'); setResult(null); setDocFile(null); setSelectedRubric(''); setDocId(''); }}
                            className="flex-1 bg-gray-200 text-gray-700 py-2 px-4 rounded hover:bg-gray-300 transition text-sm"
                        >
                            {t('evaluator.cancel') || 'Cancelar'}
                        </button>
                    </div>
                </div>
            ) : status === 'corrected' ? (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-blue-700">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">{t('evaluator.corrected') || 'Documento Corregido'}</span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                        <MarkdownTable content={correctedResult?.result || ''} />
                    </div>
                    {correctedResult?.download_url && (
                        <a href={correctedResult.download_url} download
                            className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">
                            <Download className="w-4 h-4" /> {t('evaluator.download.corrected') || 'Descargar Documento Corregido'}
                        </a>
                    )}
                    <button
                        onClick={() => { setStatus('idle'); setResult(null); setCorrectedResult(null); setDocFile(null); setSelectedRubric(''); setDocId(''); }}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700">
                        {t('evaluator.reset')}
                    </button>
                </div>
            ) : status === 'success' ? (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-green-700">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">{t('evaluator.success')}</span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                        <MarkdownTable content={result?.result || result?.content || t('evaluator.no.content')} />
                    </div>
                    {result?.download_url && (
                        <a href={result.download_url} download
                            className="inline-flex items-center gap-2 bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition">
                            <Download className="w-4 h-4" /> {t('evaluator.download.report')}
                        </a>
                    )}
                    {!result?.topic_mismatch && (
                        <button
                            onClick={handleAutoCorrect}
                            disabled={status === 'correcting'}
                            className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2 transition"
                        >
                            {status === 'correcting' ? (
                                <><Loader2 className="w-4 h-4 animate-spin" /> {t('evaluator.correcting') || 'Corrigiendo...'}</>
                            ) : (
                                <>{t('evaluator.autocorrect') || '🔧 Corregir documento automáticamente'}</>
                            )}
                        </button>
                    )}
                    <button
                        onClick={() => { setStatus('idle'); setResult(null); setCorrectedResult(null); setDocFile(null); setSelectedRubric(''); setDocId(''); }}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700">
                        {t('evaluator.reset')}
                    </button>
                </div>
            ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* Rubric dropdown from repo */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            {t('evaluator.rubric.label')}
                        </label>
                        {loadingRubrics ? (
                            <div className="flex items-center gap-2 text-sm text-gray-500 py-2">
                                <Loader2 className="w-4 h-4 animate-spin" /> {t('evaluator.loading.rubrics')}
                            </div>
                        ) : repoRubrics.length === 0 ? (
                            <p className="text-sm text-gray-500 py-2">{t('evaluator.no.rubrics')}</p>
                        ) : (
                            <select
                                value={selectedRubric}
                                onChange={(e) => setSelectedRubric(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-purple-500 focus:outline-none text-sm"
                            >
                                <option value="">{t('evaluator.rubric.select')}</option>
                                {repoRubrics.map((r) => (
                                    <option key={r.filename} value={r.filename}>
                                        {r.topics?.length > 0 ? `[${r.topics.slice(0, 3).join(', ')}] ` : ''}{r.filename}
                                    </option>
                                ))}
                            </select>
                        )}
                    </div>

                    {/* Document Upload */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            {t('evaluator.doc.label')}
                        </label>
                        <input
                            type="file"
                            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                            accept=".pdf,.docx"
                            onChange={(e) => setDocFile(e.target.files[0])}
                        />
                    </div>

                    {error && (
                        <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded">
                            <AlertCircle className="w-4 h-4" /> {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={!selectedRubric || !docFile || status === 'processing'}
                        className="w-full bg-purple-600 text-white py-2 px-4 rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
                    >
                        {status === 'processing' ? (
                            <><Loader2 className="w-4 h-4 animate-spin" /> {t('evaluator.button.evaluating')}</>
                        ) : t('evaluator.button.evaluate')}
                    </button>
                </form>
            )}
        </div>
    );
};

export default RubricEvaluator;
