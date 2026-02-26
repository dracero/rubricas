import React, { useState } from 'react';

const Orchestrator = ({ onRoute }) => {
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', text: 'Hola, soy el Orquestador del Sistema Colaba. ¿Qué deseas hacer hoy? Podes pedirme "Crear una rúbrica" o "Evaluar un documento".' }
    ]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = input;
        setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
        setInput("");
        setLoading(true);

        try {
            const res = await fetch("http://localhost:8000/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMsg })
            });

            const decision = await res.json();

            if (decision.target_agent === 'generator') {
                setMessages(prev => [...prev, { role: 'assistant', text: `Entendido. Te redirijo al Generador de Rúbricas. (${decision.reasoning})` }]);
                setTimeout(() => onRoute('generator'), 1500);
            } else if (decision.target_agent === 'evaluator') {
                setMessages(prev => [...prev, { role: 'assistant', text: `Claro. Vamos al Evaluador de Documentos. (${decision.reasoning})` }]);
                setTimeout(() => onRoute('evaluator'), 1500);
            } else {
                setMessages(prev => [...prev, { role: 'assistant', text: "No estoy seguro de qué agente necesitas. Por favor intenta ser más específico (ej: 'Evaluar tesis' o 'Crear rúbrica')." }]);
            }

        } catch (e) {
            console.error(e);
            setMessages(prev => [...prev, { role: 'assistant', text: "Error de conexión con el orquestador." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gradient-to-br from-gray-900 to-black font-sans text-white">
            <div className="w-full max-w-2xl bg-gray-800/50 backdrop-blur-xl rounded-2xl shadow-2xl border border-gray-700 overflow-hidden flex flex-col h-[600px]">
                {/* Header */}
                <div className="p-4 border-b border-gray-700 bg-black/20 flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
                    <h1 className="font-bold text-lg tracking-wide">Orquestador de Agentes</h1>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.map((m, i) => (
                        <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] p-4 rounded-2xl ${m.role === 'user'
                                    ? 'bg-blue-600 rounded-br-none text-white'
                                    : 'bg-gray-700 rounded-bl-none text-gray-200'
                                }`}>
                                {m.text}
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-gray-700 rounded-2xl rounded-bl-none p-4 flex gap-2">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-75"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Input */}
                <div className="p-4 bg-black/20 border-t border-gray-700">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Escribe tu solicitud aquí..."
                            className="flex-1 bg-gray-900 border border-gray-600 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
                        />
                        <button
                            onClick={handleSend}
                            className="bg-blue-600 hover:bg-blue-500 p-3 rounded-xl transition-colors"
                        >
                            <svg className="w-6 h-6 transform rotate-90" fill="currentColor" viewBox="0 0 20 20"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"></path></svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Orchestrator;
