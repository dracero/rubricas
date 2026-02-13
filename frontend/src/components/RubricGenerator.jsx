import React, { useState } from 'react';
import { Upload, FileText, Loader2, CheckCircle, AlertCircle, Download } from 'lucide-react';

const RubricGenerator = ({ onComplete }) => {
    const [file, setFile] = useState(null);
    const [level, setLevel] = useState('avanzado');
    const [status, setStatus] = useState('idle'); // idle, uploading, generating, success, error
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        if (e.target.files[0]) setFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;

        setStatus('uploading');
        setError('');

        try {
            // 1. Upload
            const formData = new FormData();
            formData.append('file', file);

            const uploadRes = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!uploadRes.ok) throw new Error('Error subiendo archivo');
            const uploadData = await uploadRes.json();

            // 2. Generate
            setStatus('generating');
            const genRes = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: `Generar rúbrica basada en ${file.name}`,
                    level: level,
                    document_id: uploadData.id
                })
            });

            if (!genRes.ok) throw new Error('Error generando rúbrica');
            const genData = await genRes.json();

            setResult(genData);
            setStatus('success');
            if (onComplete) onComplete(genData);

        } catch (err) {
            console.error(err);
            setError(err.message);
            setStatus('error');
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4 max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                Generador de Rúbricas
            </h3>

            {status === 'success' ? (
                <div className="space-y-4">
                    <div className="flex items-center gap-2 text-green-700">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">¡Rúbrica Generada!</span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                        <pre className="whitespace-pre-wrap text-sm text-gray-800 font-sans">
                            {result?.result || result?.content || 'Sin contenido'}
                        </pre>
                    </div>
                    {result?.download_url && (
                        <a
                            href={result.download_url}
                            download
                            className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
                        >
                            <Download className="w-4 h-4" />
                            Descargar Rúbrica
                        </a>
                    )}
                    <button
                        onClick={() => { setStatus('idle'); setResult(null); setFile(null); }}
                        className="block w-full mt-2 text-sm text-gray-500 hover:text-gray-700"
                    >
                        Generar otra
                    </button>
                </div>
            ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Documento Normativo (PDF)
                        </label>
                        <div className="flex items-center justify-center w-full">
                            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition">
                                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                    <Upload className="w-8 h-8 text-gray-400 mb-2" />
                                    <p className="text-sm text-gray-500">
                                        {file ? file.name : "Arrastra o selecciona un PDF"}
                                    </p>
                                </div>
                                <input type="file" className="hidden" accept=".pdf" onChange={handleFileChange} />
                            </label>
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Nivel Educativo</label>
                        <select
                            value={level}
                            onChange={(e) => setLevel(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
                        >
                            <option value="inicial">Inicial (Primer Año)</option>
                            <option value="avanzado">Avanzado (3°-5° año)</option>
                            <option value="posgrado">Posgrado</option>
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
                        disabled={!file || status === 'uploading' || status === 'generating'}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
                    >
                        {(status === 'uploading' || status === 'generating') ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                {status === 'uploading' ? 'Subiendo documento...' : 'Generando rúbrica (puede tardar)...'}
                            </>
                        ) : 'Generar Rúbrica'}
                    </button>
                </form>
            )}
        </div>
    );
};

export default RubricGenerator;
