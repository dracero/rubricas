import React, { useState, useEffect, useRef } from 'react';
import { FileText, Trash2, Download, Upload, Loader2, RefreshCw, FolderOpen } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatDate = (isoStr) => {
    if (!isoStr) return '—';
    try {
        return new Date(isoStr).toLocaleDateString('es-ES', {
            day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
        });
    } catch { return isoStr; }
};

const RubricRepository = () => {
    const { lang, t } = useLanguage();
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [deleting, setDeleting] = useState(null);
    const [replacing, setReplacing] = useState(null);
    const replaceInputRef = useRef(null);
    const pendingReplace = useRef(null);

    const fetchFiles = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/rubrics/files', {
                headers: { 'Accept-Language': lang },
            });
            if (!res.ok) throw new Error('Error');
            const data = await res.json();
            setFiles(data.files || []);
        } catch { setFiles([]); }
        finally { setLoading(false); }
    };

    useEffect(() => { fetchFiles(); }, []);

    const handleDelete = async (filename) => {
        if (!confirm(t('repository.confirm.delete').replace('{filename}', filename))) return;
        setDeleting(filename);
        try {
            const res = await fetch(`/api/rubrics/files/${filename}`, {
                method: 'DELETE',
                headers: { 'Accept-Language': lang },
            });
            if (res.ok) setFiles(prev => prev.filter(f => f.filename !== filename));
        } catch { /* ignore */ }
        finally { setDeleting(null); }
    };

    const handleReplaceClick = (filename) => {
        pendingReplace.current = filename;
        replaceInputRef.current?.click();
    };

    const handleReplaceFile = async (e) => {
        const file = e.target.files?.[0];
        const filename = pendingReplace.current;
        e.target.value = '';
        if (!file || !filename) return;

        setReplacing(filename);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch(`/api/rubrics/files/${filename}/replace`, {
                method: 'POST',
                headers: { 'Accept-Language': lang },
                body: formData,
            });
            if (res.ok) await fetchFiles();
        } catch { /* ignore */ }
        finally { setReplacing(null); }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-4">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                    <FolderOpen className="w-5 h-5 text-amber-600" />
                    {t('repository.title')}
                </h3>
                <button onClick={fetchFiles} className="text-gray-400 hover:text-gray-600 transition">
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            <input ref={replaceInputRef} type="file" accept=".docx" className="hidden" onChange={handleReplaceFile} />

            {loading ? (
                <div className="flex items-center gap-2 text-sm text-gray-500 py-4">
                    <Loader2 className="w-4 h-4 animate-spin" /> {t('repository.loading')}
                </div>
            ) : files.length === 0 ? (
                <p className="text-sm text-gray-500 py-4">{t('repository.empty')}</p>
            ) : (
                <ul className="space-y-2">
                    {files.map((f) => (
                        <li key={f.filename} className="flex items-center justify-between bg-gray-50 px-3 py-2 rounded border border-gray-200 text-sm">
                            <span className="flex items-center gap-2 min-w-0">
                                <FileText className="w-4 h-4 text-blue-500 shrink-0" />
                                <span className="truncate font-medium">{f.filename}</span>
                                <span className="text-gray-400 shrink-0 text-xs">
                                    {formatSize(f.size)} · {formatDate(f.created)}
                                </span>
                            </span>
                            <span className="flex items-center gap-1 shrink-0 ml-2">
                                <a href={f.download_url} download className="text-blue-500 hover:text-blue-700 p-1 transition" title={t('repository.action.download')}>
                                    <Download className="w-4 h-4" />
                                </a>
                                <button
                                    onClick={() => handleReplaceClick(f.filename)}
                                    disabled={replacing === f.filename}
                                    className="text-amber-500 hover:text-amber-700 p-1 transition disabled:opacity-50"
                                    title={t('repository.action.replace')}
                                >
                                    {replacing === f.filename ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                                </button>
                                <button
                                    onClick={() => handleDelete(f.filename)}
                                    disabled={deleting === f.filename}
                                    className="text-gray-400 hover:text-red-500 p-1 transition disabled:opacity-50"
                                    title={t('repository.action.delete')}
                                >
                                    {deleting === f.filename ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                                </button>
                            </span>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

export default RubricRepository;
