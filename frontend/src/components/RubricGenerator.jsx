import React, { useState, useCallback } from 'react';
import { FileText, Loader2, CheckCircle, AlertCircle, Download, ArrowLeft } from 'lucide-react';
import MarkdownTable from './MarkdownTable';
import MultiUploadPanel from './MultiUploadPanel';
import ExtractionProgress from './ExtractionProgress';
import ReferencePrompt from './ReferencePrompt';
import SuggestionPanel from './SuggestionPanel';

const RubricGenerator = ({ onComplete }) => {
    // Step: 'upload' | 'extracting' | 'references' | 'generate' | 'suggestions' | 'success'
    const [step, setStep] = useState('upload');
    const [batchId, setBatchId] = useState('');
    const [documentIds, setDocumentIds] = useState([]);
    const [extractionStatus, setExtractionStatus] = useState(null);
    const [level, setLevel] = useState('avanzado');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [generating, setGenerating] = useState(false);
    const [suggestions, setSuggestions] = useState([]);

    // Step 1 → 2: Upload complete
    const handleUploadComplete = useCallback((data) => {
        setBatchId(data.batch_id);
        setDocumentIds(data.accepted?.map(f => f.id) || []);
        setStep('extracting');
    }, []);

    // Step 2 → 3 or 4: Extraction complete
    const handleExtractionComplete = useCallback((statusData) => {
        setExtractionStatus(statusData);

        // Collect all references across documents
        const refs = [];
        statusData.documents?.forEach(doc => {
            if (doc.references?.length) {
                doc.references.forEach(r => {
                    refs.push({ ...r, docFilename: doc.filename });
                });
            }
        });

        if (refs.length > 0) {
            setStep('references');
        } else {
            setStep('generate');
        }
    }, []);

    // Reference uploaded → re-poll
    const handleReferenceUploaded = useCallback(() => {
        // Go back to extracting to re-poll the new document
        setStep('extracting');
    }, []);

    // Skip references → generate
    const handleSkipReferences = () => setStep('generate');

    // Step 4: Generate rubric
    const handleGenerate = async (e) => {
        e.preventDefault();
        setGenerating(true);
        setError('');

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: 'Generar rúbrica basada en los documentos subidos',
                    level,
                    document_ids: documentIds,
                }),
            });
            if (!res.ok) throw new Error('Error generando rúbrica');
            const data = await res.json();

            // Check if response has similar rubrics suggestions
            if (data.similar_rubrics && data.similar_rubrics.length > 0) {
                setSuggestions(data.similar_rubrics);
                setStep('suggestions');
            } else {
                setResult(data);
                setStep('success');
                if (onComplete) onComplete(data);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setGenerating(false);
        }
    };

    // Suggestion actions
    const handleSelectBase = async (rubricId) => {
        setGenerating(true);
        setError('');
        setSuggestions([]);

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: 'Generar rúbrica basada en los documentos subidos',
                    level,
                    document_ids: documentIds,
                    base_rubric_id: rubricId,
                    skip_search: true,
                }),
            });
            if (!res.ok) throw new Error('Error generando rúbrica');
            const data = await res.json();
            setResult(data);
            setStep('success');
            if (onComplete) onComplete(data);
        } catch (err) {
            setError(err.message);
            setStep('generate');
        } finally {
            setGenerating(false);
        }
    };

    const handleGenerateNew = async () => {
        setGenerating(true);
        setError('');
        setSuggestions([]);

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: 'Generar rúbrica basada en los documentos subidos',
                    level,
                    document_ids: documentIds,
                    skip_search: true,
                }),
            });
            if (!res.ok) throw new Error('Error generando rúbrica');
            const data = await res.json();
            setResult(data);
            setStep('success');
            if (onComplete) onComplete(data);
        } catch (err) {
            setError(err.message);
            setStep('generate');
        } finally {
            setGenerating(false);
        }
    };

    // Reset to start
    const handleReset = () => {
        setStep('upload');
        setBatchId('');
        setDocumentIds([]);
        setExtractionStatus(null);
        setResult(null);
        setError('');
        setSuggestions([]);
    };

    // Collect references for the prompt component
    const allReferences = [];
    if (extractionStatus?.documents) {
        extractionStatus.documents.forEach(doc => {
            doc.references?.forEach(r => {
                allReferences.push({ ...r, docFilename: doc.filename });
            });
        });
    }

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4 max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                Generador de Rúbricas
            </h3>

            {/* Step indicator */}
            {step !== 'upload' && step !== 'success' && (
                <div className="flex items-center gap-1 text-xs text-gray-400 mb-4">
                    <span className={step === 'upload' ? 'text-blue-600 font-medium' : 'text-green-500'}>Subir</span>
                    <span>→</span>
                    <span className={step === 'extracting' ? 'text-blue-600 font-medium' : step === 'references' || step === 'generate' || step === 'suggestions' ? 'text-green-500' : ''}>Extraer</span>
                    <span>→</span>
                    <span className={step === 'references' ? 'text-blue-600 font-medium' : step === 'generate' || step === 'suggestions' ? 'text-green-500' : ''}>Referencias</span>
                    <span>→</span>
                    <span className={step === 'generate' ? 'text-blue-600 font-medium' : step === 'suggestions' ? 'text-green-500' : ''}>Generar</span>
                    <span>→</span>
                    <span className={step === 'suggestions' ? 'text-blue-600 font-medium' : ''}>Sugerencias</span>
                </div>
            )}

            {/* Step 1: Upload */}
            {step === 'upload' && (
                <MultiUploadPanel onUploadComplete={handleUploadComplete} />
            )}

            {/* Step 2: Extraction progress */}
            {step === 'extracting' && (
                <ExtractionProgress batchId={batchId} onComplete={handleExtractionComplete} />
            )}

            {/* Step 3: References */}
            {step === 'references' && (
                <div className="space-y-4">
                    <ReferencePrompt
                        references={allReferences}
                        batchId={batchId}
                        onReferenceUploaded={handleReferenceUploaded}
                    />
                    <button
                        onClick={handleSkipReferences}
                        className="w-full text-sm text-gray-500 hover:text-gray-700 py-2 transition"
                    >
                        Continuar sin subir referencias →
                    </button>
                </div>
            )}

            {/* Step 4: Generate form */}
            {step === 'generate' && (
                <form onSubmit={handleGenerate} className="space-y-4">
                    <div className="bg-green-50 border border-green-200 rounded p-3 text-sm text-green-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 shrink-0" />
                        {documentIds.length} documento{documentIds.length > 1 ? 's' : ''} listo{documentIds.length > 1 ? 's' : ''} para generar rúbrica
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Nivel de Exigencia</label>
                        <select
                            value={level}
                            onChange={(e) => setLevel(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        >
                            <option value="inicial">Operacional (Básico)</option>
                            <option value="avanzado">Técnico/Regulatorio (Intermedio)</option>
                            <option value="critico">Alta Criticidad (Legal)</option>
                        </select>
                    </div>

                    {error && (
                        <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded">
                            <AlertCircle className="w-4 h-4" />
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={generating}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
                    >
                        {generating ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Generando rúbrica (puede tardar)...
                            </>
                        ) : 'Generar Rúbrica'}
                    </button>
                </form>
            )}

            {/* Step 4.5: Suggestions */}
            {step === 'suggestions' && (
                <div className="space-y-4">
                    {generating ? (
                        <div className="flex items-center justify-center gap-2 py-8 text-sm text-gray-500">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Generando rúbrica (puede tardar)...
                        </div>
                    ) : (
                        <>
                            {error && (
                                <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded">
                                    <AlertCircle className="w-4 h-4" />
                                    {error}
                                </div>
                            )}
                            <SuggestionPanel
                                suggestions={suggestions}
                                onSelectBase={handleSelectBase}
                                onGenerateNew={handleGenerateNew}
                                onViewFull={() => {}}
                            />
                        </>
                    )}
                </div>
            )}

            {/* Step 5: Result */}
            {step === 'success' && (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-green-700">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">¡Rúbrica Generada!</span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                        <MarkdownTable content={result?.result || result?.content || 'Sin contenido'} />
                    </div>
                    {result?.download_url && (
                        <a
                            href={result.download_url}
                            download
                            className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
                        >
                            <Download className="w-4 h-4" />
                            Descargar DOCX
                        </a>
                    )}
                    <button
                        onClick={handleReset}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700 flex items-center justify-center gap-1"
                    >
                        <ArrowLeft className="w-3.5 h-3.5" />
                        Generar otra
                    </button>
                </div>
            )}
        </div>
    );
};

export default RubricGenerator;
