import React, { useState, useEffect, useRef } from 'react';
import { Upload, Trash2, Download, FileText, X, ChevronDown, ChevronUp, Plus, Puzzle, Wrench, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLanguage } from '../contexts/LanguageContext';

const SkillManager = () => {
    const { lang, t } = useLanguage();
    const [skills, setSkills] = useState([]);
    const [availableTools, setAvailableTools] = useState([]);
    const [isOpen, setIsOpen] = useState(false);
    const [showTools, setShowTools] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState(null);
    const fileInputRef = useRef(null);

    const fetchSkills = async () => {
        try {
            const res = await fetch('/api/skills', {
                headers: { 'Accept-Language': lang },
            });
            if (res.ok) {
                const data = await res.json();
                setSkills(data.skills || []);
            }
        } catch (error) {
            console.error('Error fetching skills:', error);
        }
    };

    const fetchTools = async () => {
        try {
            const res = await fetch('/api/skills/tools', {
                headers: { 'Accept-Language': lang },
            });
            if (res.ok) {
                const data = await res.json();
                setAvailableTools(data.tools || []);
            }
        } catch (error) {
            console.error('Error fetching tools:', error);
        }
    };

    useEffect(() => {
        fetchSkills();
        fetchTools();
    }, []);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.name.endsWith('.md')) {
            setMessage({ type: 'error', text: t('skills.error.md.only') });
            return;
        }

        setUploading(true);
        setMessage(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/skills/upload', {
                method: 'POST',
                headers: { 'Accept-Language': lang },
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                setMessage({ type: 'success', text: data.message });
                await fetchSkills();
            } else {
                const err = await res.json();
                setMessage({ type: 'error', text: err.detail || t('skills.error.upload') });
            }
        } catch (error) {
            setMessage({ type: 'error', text: t('skills.error.connection') });
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const handleDelete = async (filename) => {
        if (!confirm(t('skills.confirm.delete').replace('{filename}', filename))) return;

        try {
            const res = await fetch(`/api/skills/${filename}`, {
                method: 'DELETE',
                headers: { 'Accept-Language': lang },
            });
            if (res.ok) {
                setMessage({ type: 'success', text: t('skills.deleted').replace('{filename}', filename) });
                await fetchSkills();
            } else {
                setMessage({ type: 'error', text: t('skills.error.delete') });
            }
        } catch (error) {
            setMessage({ type: 'error', text: t('skills.error.connection') });
        }
    };

    const handleDownload = (filename) => {
        window.open(`/api/skills/${filename}/download`, '_blank');
    };

    return (
        <div className="relative">
            {/* Toggle Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition-all duration-200 border border-indigo-200"
            >
                <Puzzle className="w-4 h-4" />
                <span className="text-sm font-medium">{t('skills.button')}</span>
                <span className="bg-indigo-200 text-indigo-800 text-xs font-bold px-1.5 py-0.5 rounded-full">
                    {skills.length}
                </span>
                {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {/* Dropdown Panel */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 top-full mt-2 w-96 bg-white rounded-xl shadow-2xl border border-gray-200 z-50 overflow-hidden"
                    >
                        {/* Header */}
                        <div className="px-4 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Puzzle className="w-5 h-5" />
                                <h3 className="font-semibold text-sm">{t('skills.panel.title')}</h3>
                            </div>
                            <button onClick={() => setIsOpen(false)} className="hover:bg-white/20 rounded-lg p-1 transition">
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        {/* Message */}
                        <AnimatePresence>
                            {message && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className={`px-4 py-2 text-sm ${message.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}
                                >
                                    {message.text}
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Skills List */}
                        <div className="max-h-64 overflow-y-auto divide-y divide-gray-100">
                            {skills.length === 0 ? (
                                <div className="px-4 py-8 text-center text-gray-400 text-sm">
                                    {t('skills.empty')}
                                </div>
                            ) : (
                                skills.map((skill) => (
                                    <div key={skill.filename} className="px-4 py-3 hover:bg-gray-50 transition-colors group">
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-start gap-3 flex-1 min-w-0">
                                                <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
                                                    <FileText className="w-4 h-4 text-indigo-600" />
                                                </div>
                                                <div className="min-w-0">
                                                    <p className="text-sm font-medium text-gray-800 truncate">{skill.name}</p>
                                                    <p className="text-xs text-gray-500 truncate">{skill.filename}</p>
                                                    {skill.tools && skill.tools.length > 0 && (
                                                        <div className="flex flex-wrap gap-1 mt-1">
                                                            {skill.tools.map(tl => (
                                                                <span key={tl} className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">
                                                                    {tl}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    )}
                                                    {skill.sub_agents && skill.sub_agents.length > 0 && (
                                                        <div className="flex flex-wrap gap-1 mt-1">
                                                            {skill.sub_agents.map(s => (
                                                                <span key={s} className="text-[10px] bg-purple-50 text-purple-600 px-1.5 py-0.5 rounded">
                                                                    ↳ {s}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                                                <button
                                                    onClick={() => handleDownload(skill.filename)}
                                                    className="p-1.5 rounded-lg hover:bg-blue-100 text-gray-400 hover:text-blue-600 transition"
                                                    title={t('skills.action.download')}
                                                >
                                                    <Download className="w-3.5 h-3.5" />
                                                </button>
                                                <button
                                                    onClick={() => handleDelete(skill.filename)}
                                                    className="p-1.5 rounded-lg hover:bg-red-100 text-gray-400 hover:text-red-600 transition"
                                                    title={t('skills.action.delete')}
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>

                        {/* Available Tools Documentation */}
                        <div className="border-t border-gray-200">
                            <button
                                onClick={() => setShowTools(!showTools)}
                                className="w-full px-4 py-2.5 flex items-center justify-between text-sm font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 transition-colors"
                            >
                                <div className="flex items-center gap-2">
                                    <Wrench className="w-4 h-4" />
                                    <span>{t('skills.tools.title')}</span>
                                    <span className="bg-amber-200 text-amber-800 text-xs font-bold px-1.5 py-0.5 rounded-full">
                                        {availableTools.length}
                                    </span>
                                </div>
                                {showTools ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                            </button>

                            <AnimatePresence>
                                {showTools && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className="overflow-hidden"
                                    >
                                        {/* Instructivo */}
                                        <div className="px-4 py-2 bg-blue-50 border-b border-blue-100">
                                            <div className="flex items-start gap-2">
                                                <Info className="w-3.5 h-3.5 text-blue-500 mt-0.5 shrink-0" />
                                                <p className="text-[11px] text-blue-600 leading-relaxed">
                                                    {t('skills.tools.instructions')} {t('skills.tools.example')}
                                                    <code className="block bg-blue-100 px-2 py-1 rounded mt-1 text-[10px] font-mono">
                                                        tools:<br/>
                                                        &nbsp;&nbsp;- buscar_contexto_qdrant
                                                    </code>
                                                </p>
                                            </div>
                                        </div>

                                        {/* Tools List */}
                                        <div className="max-h-48 overflow-y-auto divide-y divide-gray-100">
                                            {availableTools.map((tool) => (
                                                <div key={tool.name} className="px-4 py-3">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <Wrench className="w-3.5 h-3.5 text-amber-500" />
                                                        <code className="text-xs font-mono font-bold text-gray-800 bg-gray-100 px-1.5 py-0.5 rounded">
                                                            {tool.name}
                                                        </code>
                                                    </div>
                                                    <p className="text-[11px] text-gray-600 leading-relaxed ml-5.5 mb-1.5">
                                                        {tool.description.split('\n')[0]}
                                                    </p>
                                                    {tool.parameters && tool.parameters.length > 0 && (
                                                        <div className="ml-5.5">
                                                            <p className="text-[10px] text-gray-400 font-medium uppercase tracking-wide mb-0.5">{t('skills.tools.params')}</p>
                                                            {tool.parameters.map((param) => (
                                                                <div key={param.name} className="flex items-center gap-1.5 text-[11px]">
                                                                    <code className="text-amber-700 font-mono bg-amber-50 px-1 rounded">{param.name}</code>
                                                                    {param.type && (
                                                                        <span className="text-gray-400">({param.type})</span>
                                                                    )}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {/* Upload Button */}
                        <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".md"
                                onChange={handleUpload}
                                className="hidden"
                                id="skill-upload"
                            />
                            <label
                                htmlFor="skill-upload"
                                className={`flex items-center justify-center gap-2 w-full px-4 py-2.5 rounded-lg cursor-pointer transition-all duration-200 text-sm font-medium ${
                                    uploading
                                        ? 'bg-gray-200 text-gray-400 cursor-wait'
                                        : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm hover:shadow'
                                }`}
                            >
                                {uploading ? (
                                    <>
                                        <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                        {t('skills.upload.uploading')}
                                    </>
                                ) : (
                                    <>
                                        <Plus className="w-4 h-4" />
                                        {t('skills.upload.button')}
                                    </>
                                )}
                            </label>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default SkillManager;
