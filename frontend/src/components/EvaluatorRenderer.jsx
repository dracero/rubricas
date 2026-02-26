import React, { useState } from 'react';

const EvaluatorRenderer = ({ onBack }) => {
    const [step, setStep] = useState(1); // 1: Upload Rubric, 2: Upload Doc, 3: Result
    const [rubricId, setRubricId] = useState(null);
    const [docId, setDocId] = useState(null);
    const [evalResult, setEvalResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleRubricUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        setLoading(true);
        try {
            const res = await fetch("http://localhost:8000/api/evaluate/upload_rubric", { method: "POST", body: formData });
            const data = await res.json();
            setRubricId(data.id);
            setStep(2);
        } catch (err) {
            console.error(err);
            alert("Error subiendo rúbrica");
        } finally {
            setLoading(false);
        }
    };

    const handleDocUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        setLoading(true);
        try {
            const res = await fetch("http://localhost:8000/api/evaluate/upload_doc", { method: "POST", body: formData });
            const data = await res.json();
            setDocId(data.id);
        } catch (err) {
            console.error(err);
            alert("Error subiendo documento");
        } finally {
            setLoading(false);
        }
    };

    const handleEvaluate = async () => {
        setLoading(true);
        try {
            const res = await fetch("http://localhost:8000/api/evaluate/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ rubric_id: rubricId, doc_id: docId })
            });
            const data = await res.json();
            setEvalResult(data);
            setStep(3);
        } catch (err) {
            console.error(err);
            alert("Error en la evaluación");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6 font-sans">
            <div className="max-w-4xl mx-auto">
                <button onClick={onBack} className="mb-6 px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2">
                    ← Volver al Chat
                </button>

                <h1 className="text-3xl font-bold mb-8 bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
                    Evaluador de Documentos
                </h1>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {/* Step 1: Rubric */}
                    <div className={`p-6 rounded-xl border ${step >= 1 ? 'border-green-500 bg-green-900/10' : 'border-gray-700 bg-gray-800/50'}`}>
                        <div className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-2">Paso 1</div>
                        <h2 className="text-xl font-bold mb-4">Cargar Rúbrica</h2>
                        {rubricId ? (
                            <div className="text-green-400">✅ Rúbrica cargada</div>
                        ) : (
                            <input type="file" accept=".txt,.md" onChange={handleRubricUpload} className="text-sm text-gray-400 file:bg-gray-700 file:text-white file:border-0 file:rounded-lg file:px-3 file:py-1" />
                        )}
                    </div>

                    {/* Step 2: Document */}
                    <div className={`p-6 rounded-xl border ${step >= 2 ? 'border-blue-500 bg-blue-900/10' : 'border-gray-700 bg-gray-800/50'} ${step < 2 && 'opacity-50'}`}>
                        <div className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-2">Paso 2</div>
                        <h2 className="text-xl font-bold mb-4">Cargar Apuntes</h2>
                        {step >= 2 && (
                            docId ? (
                                <div className="text-blue-400">✅ Documento cargado</div>
                            ) : (
                                <input type="file" accept=".pdf" onChange={handleDocUpload} className="text-sm text-gray-400 file:bg-gray-700 file:text-white file:border-0 file:rounded-lg file:px-3 file:py-1" />
                            )
                        )}
                    </div>

                    {/* Step 3: Action */}
                    <div className={`p-6 rounded-xl border ${step >= 2 ? 'border-purple-500 bg-purple-900/10' : 'border-gray-700 bg-gray-800/50'} ${step < 2 && 'opacity-50'}`}>
                        <div className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-2">Paso 3</div>
                        <h2 className="text-xl font-bold mb-4">Evaluar</h2>
                        {docId && !evalResult && (
                            <button onClick={handleEvaluate} disabled={loading} className="w-full py-2 bg-white text-black font-bold rounded-lg hover:bg-gray-200">
                                {loading ? 'Evaluando...' : 'Iniciar Auditoría'}
                            </button>
                        )}
                    </div>
                </div>

                {/* Result */}
                {evalResult && (
                    <div className="mt-8 bg-gray-800 rounded-xl p-8 border border-gray-700 animate-fade-in">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-2xl font-bold">Informe de Evaluación</h2>
                            <a href={`http://localhost:8000${evalResult.download_url}`} download className="px-4 py-2 bg-green-600 rounded-lg text-sm font-bold">⬇ Descargar</a>
                        </div>
                        <div className="font-mono text-sm text-gray-300 whitespace-pre-wrap h-96 overflow-y-auto bg-black/30 p-4 rounded-lg">
                            {evalResult.content}
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
};

export default EvaluatorRenderer;
