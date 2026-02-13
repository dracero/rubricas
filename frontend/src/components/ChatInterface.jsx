import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Sparkles } from 'lucide-react';
import RubricGenerator from './RubricGenerator';
import RubricEvaluator from './RubricEvaluator';
import { clsx } from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        {
            source: 'orchestrator',
            type: 'text',
            content: 'Hola, soy RubricAI. ¿En qué puedo ayudarte hoy?',
            timestamp: new Date()
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = {
            source: 'user',
            type: 'text',
            content: input,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg.content })
            });

            if (!res.ok) throw new Error('Error de conexión');

            const data = await res.json();
            // Data format follows A2A Protocol: { source, target, type, content, metadata }

            setMessages(prev => [...prev, {
                ...data,
                timestamp: new Date()
            }]);

        } catch (error) {
            setMessages(prev => [...prev, {
                source: 'orchestrator',
                type: 'error',
                content: 'Lo siento, hubo un error de conexión con el servidor.',
                timestamp: new Date()
            }]);
        } finally {
            setLoading(false);
        }
    };

    const renderMessageContent = (msg) => {
        if (msg.type === 'action_request') {
            const componentType = msg.metadata?.component;
            return (
                <div className="space-y-4">
                    <p>{msg.content}</p>
                    {componentType === 'RubricGenerator' && (
                        <RubricGenerator />
                    )}
                    {componentType === 'RubricEvaluator' && (
                        <RubricEvaluator />
                    )}
                </div>
            );
        }
        return <p className="whitespace-pre-wrap">{msg.content}</p>;
    };

    return (
        <div className="flex flex-col h-[calc(100vh-100px)] bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6">
                <AnimatePresence>
                    {messages.map((msg, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={clsx(
                                "flex gap-3 max-w-[85%]",
                                msg.source === 'user' ? "ml-auto flex-row-reverse" : "mr-auto"
                            )}
                        >
                            <div className={clsx(
                                "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                                msg.source === 'user' ? "bg-blue-600" : "bg-blue-500"
                            )}>
                                {msg.source === 'user' ? (
                                    <User className="w-5 h-5 text-white" />
                                ) : (
                                    <Bot className="w-5 h-5 text-white" />
                                )}
                            </div>

                            <div className={clsx(
                                "p-4 rounded-2xl shadow-sm",
                                msg.source === 'user'
                                    ? "bg-blue-600 text-white rounded-tr-none"
                                    : "bg-gray-100 text-gray-800 rounded-tl-none border border-gray-200"
                            )}>
                                {renderMessageContent(msg)}
                                <span className={clsx("text-xs opacity-50 block mt-2", msg.source === 'user' ? "text-blue-100" : "text-gray-400")}>
                                    {msg.timestamp?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {loading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex gap-3 mr-auto"
                    >
                        <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                        <div className="bg-gray-100 p-4 rounded-2xl rounded-tl-none border border-gray-200 flex items-center gap-2">
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                        </div>
                    </motion.div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-gray-50 border-t border-gray-100">
                <form onSubmit={handleSend} className="max-w-4xl mx-auto relative flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Escribe un mensaje... (ej: 'Quiero crear una rúbrica')"
                        className="flex-1 p-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition shadow-sm"
                        disabled={loading}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || loading}
                        className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-sm flex items-center justify-center min-w-[50px]"
                    >
                        {loading ? <Sparkles className="w-5 h-5 animate-pulse" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatInterface;
