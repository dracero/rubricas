import React, { useState, useRef, useCallback } from 'react';
import { Upload, X, FileText, AlertCircle, Loader2 } from 'lucide-react';

const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const MultiUploadPanel = ({ batchId, onUploadComplete }) => {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [rejected, setRejected] = useState([]);
    const [error, setError] = useState('');
    const [dragOver, setDragOver] = useState(false);
    const [clearKnowledge, setClearKnowledge] = useState(true);
    const inputRef = useRef(null);

    const addFiles = useCallback((newFiles) => {
        const pdfFiles = Array.from(newFiles).filter(f =>
            f.name.toLowerCase().endsWith('.pdf')
        );
        setFiles(prev => {
            const existing = new Set(prev.map(f => f.name + f.size));
            const unique = pdfFiles.filter(f => !existing.has(f.name + f.size));
            return [...prev, ...unique];
        });
    }, []);

    const handleFileChange = (e) => {
        if (e.target.files?.length) addFiles(e.target.files);
        e.target.value = '';
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
    };

    const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
    const handleDragLeave = () => setDragOver(false);

    const removeFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleUpload = async () => {
        if (!files.length) return;
        setUploading(true);
        setError('');
        setRejected([]);

        try {
            const formData = new FormData();
            files.forEach(f => formData.append('files', f));

            const url = batchId
                ? `/api/upload/batch?batch_id=${batchId}`
                : `/api/upload/batch?clear=${clearKnowledge}`;

            const res = await fetch(url, { method: 'POST', body: formData });
            if (!res.ok) throw new Error('Error subiendo archivos');

            const data = await res.json();
            if (data.rejected?.length) setRejected(data.rejected);
            setFiles([]);
            if (onUploadComplete) onUploadComplete(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="space-y-4">
            <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => inputRef.current?.click()}
                className={`flex flex-col items-center justify-center w-full h-36 border-2 border-dashed rounded-lg cursor-pointer transition ${
                    dragOver
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-300 bg-gray-50 hover:bg-gray-100'
                }`}
            >
                <Upload className="w-8 h-8 text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">
                    Arrastra o selecciona uno o más PDFs
                </p>
                <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept=".pdf"
                    className="hidden"
                    onChange={handleFileChange}
                />
            </div>

            {files.length > 0 && (
                <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-700">
                        {files.length} archivo{files.length > 1 ? 's' : ''} seleccionado{files.length > 1 ? 's' : ''}
                    </p>
                    <ul className="space-y-1">
                        {files.map((f, i) => (
                            <li key={i} className="flex items-center justify-between bg-gray-50 px-3 py-2 rounded border border-gray-200 text-sm">
                                <span className="flex items-center gap-2 truncate">
                                    <FileText className="w-4 h-4 text-blue-500 shrink-0" />
                                    <span className="truncate">{f.name}</span>
                                    <span className="text-gray-400 shrink-0">({formatSize(f.size)})</span>
                                </span>
                                <button onClick={() => removeFile(i)} className="text-gray-400 hover:text-red-500 ml-2 shrink-0">
                                    <X className="w-4 h-4" />
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {error && (
                <div className="flex items-center gap-2 p-3 text-sm text-red-700 bg-red-50 rounded">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    {error}
                </div>
            )}

            {rejected.length > 0 && (
                <div className="p-3 bg-yellow-50 rounded border border-yellow-200 text-sm space-y-1">
                    <p className="font-medium text-yellow-800 flex items-center gap-1">
                        <AlertCircle className="w-4 h-4" /> Archivos rechazados:
                    </p>
                    {rejected.map((r, i) => (
                        <p key={i} className="text-yellow-700 ml-5">
                            {r.filename} — {r.reason}
                        </p>
                    ))}
                </div>
            )}

            {!batchId && (
                <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
                    <input
                        type="checkbox"
                        checked={clearKnowledge}
                        onChange={(e) => setClearKnowledge(e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    Limpiar base de conocimiento (recomendado para nueva normativa)
                </label>
            )}

            <button
                onClick={handleUpload}
                disabled={!files.length || uploading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
            >
                {uploading ? (
                    <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Subiendo documentos...
                    </>
                ) : (
                    <>
                        <Upload className="w-4 h-4" />
                        Subir documentos
                    </>
                )}
            </button>
        </div>
    );
};

export default MultiUploadPanel;
