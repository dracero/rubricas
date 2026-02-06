import React, { useState } from 'react';
import { Upload, FileSearch, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

const RubricEvaluator = ({ onComplete }) => {
    const [rubricFile, setRubricFile] = useState(null);
    const [docFile, setDocFile] = useState(null);
    const [status, setStatus] = useState('idle');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!rubricFile || !docFile) return;

        setStatus('processing');
        setError('');

        try {
            // 1. Upload Rubric
            const formRubric = new FormData();
            formRubric.append('file', rubricFile);
            const rubRes = await fetch('http://localhost:8000/api/evaluate/upload_rubric', {
                method: 'POST',
                body: formRubric
            });
            if (!rubRes.ok) throw new Error('Error subiendo rúbrica');
            const rubData = await rubRes.json();

            // 2. Upload Doc
            const formDoc = new FormData();
            formDoc.append('file', docFile);
            const docRes = await fetch('http://localhost:8000/api/evaluate/upload_doc', {
                method: 'POST',
                body: formDoc
            });
            if (!docRes.ok) throw new Error('Error subiendo documento');
            const docData = await docRes.json();

            // 3. Evaluate
            const evalRes = await fetch('http://localhost:8000/api/evaluate/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rubric_id: rubData.id,
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
                Evaluador de Apuntes
            </h3>

            {status === 'success' ? (
                <div className="text-center p-4">
                    <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                    <p className="text-green-700 font-medium">¡Evaluación Completada!</p>
                    <a
                        href={`http://localhost:8000${result.download_url}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-block mt-4 bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition"
                    >
                        Descargar Informe
                    </a>
                    <button
                        onClick={() => setStatus('idle')}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700"
                    >
                        Evaluar otro
                    </button>
                </div>
            ) : (
                <form onSubmit={handleSubmit} className="space-y-4">

                    {/* Rubric Upload */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Rúbrica de Referencia (.txt/.md)
                        </label>
                        <div className="flex items-center gap-2">
                            <input
                                type="file"
                                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                                accept=".txt,.md"
                                onChange={(e) => setRubricFile(e.target.files[0])}
                            />
                        </div>
                    </div>

                    {/* Student Doc Upload */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Apuntes del Estudiante (.pdf)
                        </label>
                        <div className="flex items-center gap-2">
                            <input
                                type="file"
                                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                                accept=".pdf"
                                onChange={(e) => setDocFile(e.target.files[0])}
                            />
                        </div>
                    </div>

                    {error && (
                        <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded">
                            <AlertCircle className="w-4 h-4" />
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={!rubricFile || !docFile || status === 'processing'}
                        className="w-full bg-purple-600 text-white py-2 px-4 rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
                    >
                        {status === 'processing' ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Analizando...
                            </>
                        ) : 'Iniciar Evaluación'}
                    </button>
                </form>
            )}
        </div>
    );
};

export default RubricEvaluator;
