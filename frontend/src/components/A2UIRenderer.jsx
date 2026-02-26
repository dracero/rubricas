import React, { useState } from 'react';

const A2UIRenderer = () => {
    const [file, setFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState("idle"); // idle, uploading, success, error
    const [uploadResult, setUploadResult] = useState(null);

    const [prompt, setPrompt] = useState("");
    const [level, setLevel] = useState("avanzado");
    const [generationStatus, setGenerationStatus] = useState("idle"); // idle, generating, success, error
    const [rubricContent, setRubricContent] = useState("");
    const [downloadUrl, setDownloadUrl] = useState("");

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploadStatus("uploading");

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("http://localhost:8000/api/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Upload failed");

            const data = await res.json();
            setUploadResult(data);
            setUploadStatus("success");
            setPrompt(`Generar rúbrica basada en ${file.name}`);
        } catch (error) {
            console.error(error);
            setUploadStatus("error");
        }
    };

    const handleGenerate = async () => {
        setGenerationStatus("generating");

        try {
            const res = await fetch("http://localhost:8000/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: prompt,
                    level: level
                })
            });

            if (!res.ok) throw new Error("Generation failed");

            const data = await res.json();
            setRubricContent(data.content);
            setDownloadUrl(`http://localhost:8000${data.download_url}`);
            setGenerationStatus("success");
        } catch (error) {
            console.error(error);
            setGenerationStatus("error");
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-gray-50 font-sans">
            <div className="w-full max-w-5xl bg-white rounded-2xl shadow-2xl overflow-hidden grid grid-cols-1 md:grid-cols-2">

                {/* Left Column: Visual/Context */}
                <div className="hidden md:flex flex-col justify-center p-12 bg-gradient-to-br from-blue-600 to-indigo-700 text-white relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none">
                        {/* Abstract Pattern */}
                        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
                            <path d="M0 100 C 20 0 50 0 100 100 Z" fill="white" />
                        </svg>
                    </div>

                    <div className="relative z-10">
                        <h1 className="text-4xl font-bold mb-4">Generador de Rúbricas</h1>
                        <p className="text-blue-100 text-lg mb-8">
                            Transforma tus documentos normativos en herramientas de evaluación precisas en segundos.
                        </p>

                        <div className="space-y-4">
                            <div className="flex items-center gap-3 bg-white/10 p-3 rounded-lg backdrop-blur-sm">
                                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">1</div>
                                <span>Sube tu PDF normativo</span>
                            </div>
                            <div className="flex items-center gap-3 bg-white/10 p-3 rounded-lg backdrop-blur-sm">
                                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">2</div>
                                <span>Define el contexto</span>
                            </div>
                            <div className="flex items-center gap-3 bg-white/10 p-3 rounded-lg backdrop-blur-sm">
                                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">3</div>
                                <span>Obtén tu rúbrica lista</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column: Form */}
                <div className="p-8 md:p-12 space-y-8 overflow-y-auto max-h-[90vh]">

                    {/* Header Mobile Only */}
                    <div className="md:hidden text-center mb-6">
                        <h1 className="text-3xl font-bold text-gray-900">Generador de Rúbricas</h1>
                    </div>

                    {/* Section 1: Upload */}
                    <section>
                        <h2 className="text-sm uppercase tracking-wider text-gray-500 font-bold mb-4">1. Documento Base</h2>
                        <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-400'
                            }`}>
                            {!file ? (
                                <>
                                    <div className="mx-auto w-12 h-12 text-gray-300 mb-3">
                                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                                    </div>
                                    <label htmlFor="file-upload" className="cursor-pointer">
                                        <span className="text-blue-600 font-medium hover:text-blue-700">Explora tus archivos</span>
                                        <span className="text-gray-500 block text-sm mt-1">o arrastra un PDF aquí</span>
                                    </label>
                                    <input id="file-upload" type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
                                </>
                            ) : (
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3 text-left">
                                        <div className="bg-white p-2 rounded-lg shadow-sm">
                                            <span className="text-2xl">📄</span>
                                        </div>
                                        <div>
                                            <p className="font-medium text-gray-900 truncate max-w-[200px]">{file.name}</p>
                                            <p className="text-xs text-gray-500">Listo para subir</p>
                                        </div>
                                    </div>
                                    <button onClick={() => setFile(null)} className="text-gray-400 hover:text-red-500 p-2">✕</button>
                                </div>
                            )}
                        </div>

                        {file && (
                            <button
                                onClick={handleUpload}
                                disabled={uploadStatus === 'uploading' || uploadStatus === 'success'}
                                className={`w-full mt-4 py-3 rounded-lg font-semibold transition-all shadow-sm ${uploadStatus === 'success' ? 'bg-green-100 text-green-700' : 'bg-gray-900 text-white hover:bg-black'
                                    }`}
                            >
                                {uploadStatus === 'uploading' ? 'Analizando...' : uploadStatus === 'success' ? 'Documento Procesado Correctamente' : 'Subir y Procesar'}
                            </button>
                        )}
                    </section>

                    {/* Section 2: Details */}
                    <section className={`transition-opacity duration-500 ${uploadStatus !== 'success' ? 'opacity-40 pointer-events-none' : 'opacity-100'}`}>
                        <h2 className="text-sm uppercase tracking-wider text-gray-500 font-bold mb-4">2. Configuración</h2>

                        <div className="space-y-5">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Tema o Enfoque</label>
                                <input
                                    type="text"
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    placeholder="Ej: Rúbrica para evaluar el Taller de Tesis..."
                                    className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Nivel Académico</label>
                                <div className="grid grid-cols-3 gap-3">
                                    {['primer_año', 'avanzado', 'posgrado'].map((lvl) => (
                                        <button
                                            key={lvl}
                                            onClick={() => setLevel(lvl)}
                                            className={`py-3 px-2 rounded-lg text-sm font-medium border transition-all ${level === lvl
                                                    ? 'bg-blue-50 border-blue-500 text-blue-700'
                                                    : 'border-gray-200 text-gray-600 hover:bg-gray-50'
                                                }`}
                                        >
                                            {lvl.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={handleGenerate}
                                disabled={generationStatus === 'generating'}
                                className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-lg font-bold rounded-xl hover:shadow-lg hover:scale-[1.01] transition-all active:scale-[0.99] flex justify-center items-center gap-2"
                            >
                                {generationStatus === 'generating' ? (
                                    <>
                                        <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                        Generando...
                                    </>
                                ) : '✨ Generar Rúbrica'}
                            </button>
                        </div>
                    </section>

                    {/* Section 3: Output */}
                    {generationStatus === 'success' && (
                        <div className="animate-fade-in-up">
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="text-sm uppercase tracking-wider text-green-600 font-bold">Resultado Listo</h2>
                                <a href={downloadUrl} className="text-blue-600 text-sm font-medium hover:underline flex items-center gap-1">
                                    Descargar Archivo <span>⬇</span>
                                </a>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-6 border border-gray-100 h-64 overflow-y-auto text-xs font-mono text-gray-600 shadow-inner">
                                {rubricContent}
                            </div>
                        </div>
                    )}

                </div>
            </div>
        </div>
    );
};

export default A2UIRenderer;
