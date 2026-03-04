import React, { useState, useEffect } from 'react';
import { Settings, ChevronDown, Eye, EyeOff, Sparkles, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

const WelcomeSetup = ({ onComplete }) => {
    const [catalog, setCatalog] = useState({});
    const [provider, setProvider] = useState('');
    const [modelId, setModelId] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [showKey, setShowKey] = useState(false);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        fetch('/api/llm/catalog')
            .then(res => res.json())
            .then(data => {
                setCatalog(data);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, []);

    const handleProviderChange = (e) => {
        const newProvider = e.target.value;
        setProvider(newProvider);
        setModelId(''); // Reset model when provider changes
    };

    const currentModels = provider && catalog[provider]
        ? catalog[provider].models
        : [];

    const handleContinue = async (useDefaults = false) => {
        if (useDefaults) {
            onComplete();
            return;
        }

        // Only send non-empty values
        const config = {};
        if (provider) config.provider = provider;
        if (modelId) config.model_id = modelId;
        if (apiKey) config.api_key = apiKey;

        if (Object.keys(config).length > 0) {
            setSaving(true);
            try {
                await fetch('/api/llm/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config),
                });
            } catch (err) {
                console.error('Error saving LLM config:', err);
            }
            setSaving(false);
        }

        onComplete();
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
                <div className="animate-pulse text-blue-600 text-lg">Cargando configuración...</div>
            </div>
        );
    }

    return (
        <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="w-full max-w-lg mx-4"
            >
                {/* Header */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 mb-4 shadow-lg">
                        <Sparkles className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-800">RubricAI</h1>
                    <p className="text-gray-500 mt-2">Sistema de Rúbricas con IA</p>
                </div>

                {/* Config Card */}
                <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 space-y-5">
                    <div className="flex items-center gap-2 text-gray-700">
                        <Settings className="w-5 h-5 text-blue-600" />
                        <h2 className="font-semibold text-lg">Configuración del Modelo</h2>
                    </div>

                    <p className="text-sm text-gray-500">
                        Seleccioná el proveedor y modelo de IA, o dejá en blanco para usar la configuración por defecto.
                    </p>

                    {/* Provider */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Proveedor
                        </label>
                        <div className="relative">
                            <select
                                value={provider}
                                onChange={handleProviderChange}
                                className="w-full p-3 pr-10 border border-gray-300 rounded-xl bg-white focus:ring-2 focus:ring-blue-500 focus:outline-none appearance-none transition"
                            >
                                <option value="">— Por defecto (Groq) —</option>
                                {Object.entries(catalog).map(([key, val]) => (
                                    <option key={key} value={key}>{val.name}</option>
                                ))}
                            </select>
                            <ChevronDown className="w-4 h-4 text-gray-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                        </div>
                    </div>

                    {/* Model */}
                    {provider && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            transition={{ duration: 0.2 }}
                        >
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Modelo
                            </label>
                            <div className="relative">
                                <select
                                    value={modelId}
                                    onChange={(e) => setModelId(e.target.value)}
                                    className="w-full p-3 pr-10 border border-gray-300 rounded-xl bg-white focus:ring-2 focus:ring-blue-500 focus:outline-none appearance-none transition"
                                >
                                    <option value="">— Modelo por defecto —</option>
                                    {currentModels.map((m) => (
                                        <option key={m.id} value={m.id}>{m.name}</option>
                                    ))}
                                </select>
                                <ChevronDown className="w-4 h-4 text-gray-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                            </div>
                        </motion.div>
                    )}

                    {/* API Key */}
                    {provider && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            transition={{ duration: 0.2, delay: 0.1 }}
                        >
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                API Key <span className="text-gray-400 font-normal">(opcional)</span>
                            </label>
                            <div className="relative">
                                <input
                                    type={showKey ? 'text' : 'password'}
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    placeholder="Dejá vacío para usar la del servidor"
                                    className="w-full p-3 pr-10 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none transition"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowKey(!showKey)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                >
                                    {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                </button>
                            </div>
                        </motion.div>
                    )}

                    {/* Buttons */}
                    <div className="flex gap-3 pt-2">
                        <button
                            onClick={() => handleContinue(true)}
                            className="flex-1 py-3 px-4 text-gray-600 bg-gray-100 rounded-xl hover:bg-gray-200 transition font-medium text-sm"
                            disabled={saving}
                        >
                            Usar por defecto
                        </button>
                        <button
                            onClick={() => handleContinue(false)}
                            disabled={saving}
                            className="flex-1 py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition font-medium text-sm flex items-center justify-center gap-2 shadow-md disabled:opacity-50"
                        >
                            {saving ? 'Guardando...' : (
                                <>Continuar <ArrowRight className="w-4 h-4" /></>
                            )}
                        </button>
                    </div>
                </div>

                <p className="text-center text-xs text-gray-400 mt-4">
                    Podés cambiar la configuración en cualquier momento desde el ícono ⚙️
                </p>
            </motion.div>
        </div>
    );
};

export default WelcomeSetup;
