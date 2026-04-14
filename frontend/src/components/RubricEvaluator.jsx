import React, { useState, useEffect } from 'react';
import { Upload, FileSearch, Loader2, CheckCircle, AlertCircle, Download } from 'lucide-react';
import MarkdownTable from './MarkdownTable';

const RubricEvaluator = ({ onComplete }) => {
    const [repoRubrics, setRepoRubrics] = useState([]);
    const [selectedRubric, setSelectedRubric] = useState('');
    const [docFile, setDocFile] = useState(null);
    const [status, setStatus] = useState('idle');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [loadingRubrics, setLoadingRubrics] = useState(true);

    // Load rubrics from repo on mount
    useEffect(() => {
        const fetchRubrics = async () => {
            try {
                const res = await fetch('/api/rubrics/files');
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
                body: formDoc
            });
            if (!docRes.ok) throw new Error('Error subiendo documento');
            const docData = await docRes.json();

            // 2. Evaluate using repo rubric filename
            const evalRes = await fetch('/api/evaluate/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rubric_filename: selectedRubric,
                    doc_id: docData.id
                })
            });

            if (!evalRes.ok) throw new Error('Error en evaluación');
            const evalData = await evalRes.json();

            setResult(evalData);
            setStatus('success');
            if (onComplete) onComplete(evalData);

        } catch (err) {
            console.error(err);
            setError(err.message);
            setStatus('error');
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4 max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FileSearch className="w-5 h-5 text-purple-600" />
                Evaluador de Documentos
            </h3>

            {status === 'success' ? (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-green-700">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">¡Evaluación Completada!</span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                        <MarkdownTable content={result?.result || result?.content || 'Sin contenido'} />
                    </div>
                    {result?.download_url && (
                        <a href={result.download_url} download
                            className="inline-flex items-center gap-2 bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition">
                            <Download className="w-4 h-4" /> Descargar Informe
                        </a>
                    )}
                    <button
                        onClick={() => { setStatus('idle'); setResult(null); setDocFile(null); setSelectedRubric(''); }}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700">
                        Evaluar otro
                    </button>
                </div>
            ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* Rubric dropdown from repo */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Rúbrica de Referencia
                        </label>
                        {loadingRubrics ? (
                            <div className="flex items-center gap-2 text-sm text-gray-500 py-2">
                                <Loader2 className="w-4 h-4 animate-spin" /> Cargando rúbricas...
                            </div>
                        ) : repoRubrics.length === 0 ? (
                            <p className="text-sm text-gray-500 py-2">No hay rúbricas en el repositorio. Generá una primero.</p>
                        ) : (
                            <select
                                value={selectedRubric}
                                onChange={(e) => setSelectedRubric(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-purple-500 focus:outline-none text-sm"
                            >
                                <option value="">Seleccionar rúbrica...</option>
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
                            Documento a Evaluar (.pdf/.docx)
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
                            <><Loader2 className="w-4 h-4 animate-spin" /> Analizando...</>
                        ) : 'Iniciar Evaluación'}
                    </button>
                </form>
            )}
        </div>
    );
};

export default RubricEvaluator;
