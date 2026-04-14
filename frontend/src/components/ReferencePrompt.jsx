import React, { useRef, useState } from 'react';
import { FileText, Upload, Loader2, ExternalLink } from 'lucide-react';

const typeBadge = (type) => {
    const colors = {
        normativa: 'bg-purple-100 text-purple-700',
        url: 'bg-blue-100 text-blue-700',
        codigo_referencia: 'bg-amber-100 text-amber-700',
    };
    return colors[type] || 'bg-gray-100 text-gray-700';
};

const ReferencePrompt = ({ references, batchId, onReferenceUploaded }) => {
    const [uploadingRef, setUploadingRef] = useState(null);
    const inputRef = useRef(null);
    const pendingRefIndex = useRef(null);

    // Group references by source document
    // references comes from statusData.documents — each doc has .references[]
    // We expect the parent to pass an array of { docFilename, refs: [{type, text, url}] }
    // OR a flat array; we handle both.
    if (!references || references.length === 0) return null;

    // Normalize: if flat array of {type, text, url, docFilename?}, group by docFilename
    const grouped = {};
    references.forEach(r => {
        const key = r.docFilename || 'Documento';
        if (!grouped[key]) grouped[key] = [];
        grouped[key].push(r);
    });

    const groups = Object.entries(grouped);
    if (groups.length === 0) return null;

    const handleUploadClick = (groupIdx, refIdx) => {
        pendingRefIndex.current = `${groupIdx}-${refIdx}`;
        inputRef.current?.click();
    };

    const handleFileSelected = async (e) => {
        const file = e.target.files?.[0];
        if (!file || !batchId) return;
        e.target.value = '';

        setUploadingRef(pendingRefIndex.current);
        try {
            const formData = new FormData();
            formData.append('files', file);
            const res = await fetch(`/api/upload/batch?batch_id=${batchId}`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) throw new Error('Error subiendo documento referenciado');
            if (onReferenceUploaded) onReferenceUploaded();
        } catch (err) {
            console.error(err);
        } finally {
            setUploadingRef(null);
        }
    };

    return (
        <div className="space-y-3">
            <p className="text-sm font-medium text-gray-700">
                Referencias detectadas
            </p>
            <p className="text-xs text-gray-500">
                Se encontraron referencias a documentos externos. Si dispones de alguno, puedes subirlo para enriquecer la ontología.
            </p>

            <input
                ref={inputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handleFileSelected}
            />

            {groups.map(([docName, refs], gi) => (
                <div key={gi} className="bg-gray-50 rounded border border-gray-200 p-3 space-y-2">
                    <p className="text-xs font-medium text-gray-600 flex items-center gap-1">
                        <FileText className="w-3.5 h-3.5" />
                        {docName}
                    </p>
                    <ul className="space-y-1.5">
                        {refs.map((ref, ri) => (
                            <li key={ri} className="flex items-center justify-between gap-2 text-sm">
                                <span className="flex items-center gap-2 min-w-0">
                                    <span className={`text-xs px-1.5 py-0.5 rounded font-medium shrink-0 ${typeBadge(ref.type)}`}>
                                        {ref.type}
                                    </span>
                                    <span className="truncate">{ref.text}</span>
                                    {ref.url && (
                                        <a href={ref.url} target="_blank" rel="noopener noreferrer" className="text-blue-500 shrink-0">
                                            <ExternalLink className="w-3.5 h-3.5" />
                                        </a>
                                    )}
                                </span>
                                <button
                                    onClick={() => handleUploadClick(gi, ri)}
                                    disabled={uploadingRef === `${gi}-${ri}`}
                                    className="text-xs text-blue-600 hover:text-blue-800 whitespace-nowrap flex items-center gap-1 shrink-0 disabled:opacity-50"
                                >
                                    {uploadingRef === `${gi}-${ri}` ? (
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                    ) : (
                                        <Upload className="w-3 h-3" />
                                    )}
                                    Tengo este documento
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
};

export default ReferencePrompt;
