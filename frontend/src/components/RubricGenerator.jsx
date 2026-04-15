import React, { useState, useCallback, useEffect } from 'react';
import { FileText, Loader2, CheckCircle, AlertCircle, Download, ArrowLeft, FolderOpen, Plus } from 'lucide-react';
import MarkdownTable from './MarkdownTable';
import MultiUploadPanel from './MultiUploadPanel';
import ExtractionProgress from './ExtractionProgress';
import ReferencePrompt from './ReferencePrompt';
import { useLanguage } from '../contexts/LanguageContext';

const RubricGenerator = ({ onComplete }) => {
    const { lang, t } = useLanguage();
    // Step: 'existing' | 'upload' | 'extracting' | 'references' | 'generate' | 'suggestions' | 'success'
    const [step, setStep] = useState('existing');
    const [batchId, setBatchId] = useState('');
    const [documentIds, setDocumentIds] = useState([]);
    const [extractionStatus, setExtractionStatus] = useState(null);
    const [level, setLevel] = useState('avanzado');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [generating, setGenerating] = useState(false);
    const [suggestions, setSuggestions] = useState([]);
    const [similarNames, setSimilarNames] = useState([]);
    const [existingRubrics, setExistingRubrics] = useState([]);
    const [loadingExisting, setLoadingExisting] = useState(true);

    // Load existing rubrics on mount
    useEffect(() => {
        const fetchExisting = async () => {
            setLoadingExisting(true);
            try {
                const res = await fetch('/api/rubrics/files', {
                    headers: { 'Accept-Language': lang },
                });
                if (res.ok) {
                    const data = await res.json();
                    setExistingRubrics(data.files || []);
                }
            } catch { /* ignore */ }
            finally { setLoadingExisting(false); }
        };
        fetchExisting();
    }, []);

    const handleUploadComplete = useCallback((data) => {
        setBatchId(data.batch_id);
        setDocumentIds(data.accepted?.map(f => f.id) || []);
        setStep('extracting');
    }, []);

    const handleExtractionComplete = useCallback((statusData) => {
        setExtractionStatus(statusData);
        const refs = [];
        statusData.documents?.forEach(doc => {
            if (doc.references?.length) {
                doc.references.forEach(r => refs.push({ ...r, docFilename: doc.filename }));
            }
        });
        setStep(refs.length > 0 ? 'references' : 'generate');
    }, []);

    const handleReferenceUploaded = useCallback(() => setStep('extracting'), []);
    const handleSkipReferences = () => setStep('generate');

    const handleGenerate = async (e) => {
        e.preventDefault();
        setGenerating(true);
        setError('');
        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept-Language': lang,
                },
                body: JSON.stringify({ prompt: 'Generar rúbrica', level, document_ids: documentIds }),
            });
            if (!res.ok) throw new Error(t('generator.error.generate'));
            const data = await res.json();
            if (data.similar_rubrics?.length > 0) {
                setSuggestions(data.similar_rubrics);
                setSimilarNames(data.similar_names || []);
                setStep('suggestions');
            } else {
                setResult(data);
                setStep('success');
                if (onComplete) onComplete(data);
            }
        } catch (err) { setError(err.message); }
        finally { setGenerating(false); }
    };

    const handleSelectBase = async (rubricId) => {
        setGenerating(true);
        setError('');
        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept-Language': lang,
                },
                body: JSON.stringify({ prompt: 'Generar rúbrica', level, document_ids: documentIds, base_rubric_id: rubricId, skip_search: true }),
            });
            if (!res.ok) throw new Error(t('generator.error.generate'));
            const data = await res.json();
            setResult(data);
            setStep('success');
            if (onComplete) onComplete(data);
        } catch (err) { setError(err.message); setStep('generate'); }
        finally { setGenerating(false); }
    };

    const handleGenerateNew = async () => {
        setGenerating(true);
        setError('');
        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept-Language': lang,
                },
                body: JSON.stringify({ prompt: 'Generar rúbrica', level, document_ids: documentIds, skip_search: true }),
            });
            if (!res.ok) throw new Error(t('generator.error.generate'));
            const data = await res.json();
            setResult(data);
            setStep('success');
            if (onComplete) onComplete(data);
        } catch (err) { setError(err.message); setStep('generate'); }
        finally { setGenerating(false); }
    };

    const handleReset = () => {
        setStep('existing');
        setBatchId(''); setDocumentIds([]); setExtractionStatus(null);
        setResult(null); setError(''); setSuggestions([]); setSimilarNames([]);
    };

    const allReferences = [];
    if (extractionStatus?.documents) {
        extractionStatus.documents.forEach(doc => {
            doc.references?.forEach(r => allReferences.push({ ...r, docFilename: doc.filename }));
        });
    }

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4 max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                {t('generator.title')}
            </h3>

            {/* Step 0: Show existing rubrics first */}
            {step === 'existing' && (
                <div className="space-y-4">
                    {loadingExisting ? (
                        <div className="flex items-center gap-2 text-sm text-gray-500 py-4">
                            <Loader2 className="w-4 h-4 animate-spin" /> {t('generator.loading.existing')}
                        </div>
                    ) : existingRubrics.length > 0 ? (
                        <>
                            <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm text-blue-700 flex items-center gap-2">
                                <FolderOpen className="w-4 h-4 shrink-0" />
                                {t('generator.existing.hint')}
                            </div>
                            <ul className="space-y-2">
                                {existingRubrics.map((f) => (
                                    <li key={f.filename} className="bg-gray-50 px-3 py-2 rounded border border-gray-200 text-sm space-y-1">
                                        {f.topics?.length > 0 && (
                                            <div className="flex flex-wrap gap-1">
                                                {f.topics.map((tp, i) => (
                                                    <span key={i} className="text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">{tp}</span>
                                                ))}
                                            </div>
                                        )}
                                        <div className="flex items-center justify-between">
                                            <span className="flex items-center gap-2 min-w-0">
                                                <FileText className="w-4 h-4 text-blue-500 shrink-0" />
                                                <span className="truncate font-medium">{f.filename}</span>
                                            </span>
                                            <a href={f.download_url} download className="text-xs text-blue-600 hover:text-blue-800 shrink-0 ml-2">
                                                <Download className="w-4 h-4" />
                                            </a>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </>
                    ) : (
                        <p className="text-sm text-gray-500">{t('generator.no.rubrics')}</p>
                    )}
                    <button
                        onClick={() => setStep('upload')}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition text-sm flex items-center justify-center gap-2"
                    >
                        <Plus className="w-4 h-4" />
                        {t('generator.new')}
                    </button>
                </div>
            )}

            {/* Step 1: Upload */}
            {step === 'upload' && (
                <MultiUploadPanel onUploadComplete={handleUploadComplete} />
            )}

            {/* Step 2: Extraction */}
            {step === 'extracting' && (
                <ExtractionProgress batchId={batchId} onComplete={handleExtractionComplete} />
            )}

            {/* Step 3: References */}
            {step === 'references' && (
                <div className="space-y-4">
                    <ReferencePrompt references={allReferences} batchId={batchId} onReferenceUploaded={handleReferenceUploaded} />
                    <button onClick={handleSkipReferences} className="w-full text-sm text-gray-500 hover:text-gray-700 py-2 transition">
                        {t('generator.skip.references')}
                    </button>
                </div>
            )}

            {/* Step 4: Generate form */}
            {step === 'generate' && (
                <form onSubmit={handleGenerate} className="space-y-4">
                    <div className="bg-green-50 border border-green-200 rounded p-3 text-sm text-green-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 shrink-0" />
                        {documentIds.length} {documentIds.length > 1 ? t('generator.docs.ready.plural') : t('generator.docs.ready.singular')}
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">{t('generator.level.label')}</label>
                        <select value={level} onChange={(e) => setLevel(e.target.value)} className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:outline-none">
                            <option value="inicial">{t('generator.level.inicial')}</option>
                            <option value="avanzado">{t('generator.level.avanzado')}</option>
                            <option value="critico">{t('generator.level.critico')}</option>
                        </select>
                    </div>
                    {error && <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded"><AlertCircle className="w-4 h-4" />{error}</div>}
                    <button type="submit" disabled={generating} className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition">
                        {generating ? <><Loader2 className="w-4 h-4 animate-spin" />{t('generator.button.generating')}</> : t('generator.button.generate')}
                    </button>
                </form>
            )}

            {/* Step 4.5: Suggestions */}
            {step === 'suggestions' && (
                <div className="space-y-4">
                    {generating ? (
                        <div className="flex items-center justify-center gap-2 py-8 text-sm text-gray-500">
                            <Loader2 className="w-4 h-4 animate-spin" />{t('generator.button.generating')}
                        </div>
                    ) : (
                        <>
                            {error && <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded"><AlertCircle className="w-4 h-4" />{error}</div>}
                            <div className="bg-amber-50 border border-amber-200 rounded p-4 text-sm">
                                <p className="font-medium text-amber-800 mb-2">{t('generator.similar.found')}</p>
                                <ul className="space-y-1">
                                    {similarNames.map((s, i) => (
                                        <li key={i} className="flex items-center justify-between text-amber-700">
                                            <span className="flex items-center gap-2">
                                                <FileText className="w-4 h-4 shrink-0" />
                                                <span className="font-medium">{s.filename}</span>
                                                <span className="text-xs text-amber-500">({s.score}%)</span>
                                            </span>
                                            <button onClick={() => handleSelectBase(s.rubric_id)} className="text-xs text-blue-600 hover:text-blue-800 underline">{t('generator.similar.use')}</button>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <button onClick={handleGenerateNew} className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition text-sm">
                                {t('generator.similar.generate.new')}
                            </button>
                        </>
                    )}
                </div>
            )}

            {/* Step 5: Result */}
            {step === 'success' && (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-green-700"><CheckCircle className="w-5 h-5" /><span className="font-medium">{t('generator.success')}</span></div>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                        <MarkdownTable content={result?.result || result?.content || t('generator.no.content')} />
                    </div>
                    {result?.download_url && (
                        <a href={result.download_url} download className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">
                            <Download className="w-4 h-4" />{t('generator.download.docx')}
                        </a>
                    )}
                    <button onClick={handleReset} className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700 flex items-center justify-center gap-1">
                        <ArrowLeft className="w-3.5 h-3.5" />{t('generator.reset')}
                    </button>
                </div>
            )}
        </div>
    );
};

export default RubricGenerator;
