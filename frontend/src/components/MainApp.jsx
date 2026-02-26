import React, { useState } from 'react';
import Orchestrator from './Orchestrator';
import A2UIRenderer from './A2UIRenderer';
import EvaluatorRenderer from './EvaluatorRenderer';

const MainApp = () => {
    const [view, setView] = useState('chat'); // chat, generator, evaluator

    return (
        <>
            {view === 'chat' && (
                <Orchestrator onRoute={(route) => setView(route)} />
            )}

            {view === 'generator' && (
                <div className="relative">
                    <button
                        onClick={() => setView('chat')}
                        className="absolute top-4 left-4 z-50 px-4 py-2 bg-gray-900 text-white rounded-full text-sm font-medium shadow-lg hover:bg-black transition-all"
                    >
                        ← Volver al Orquestador
                    </button>
                    <A2UIRenderer />
                </div>
            )}

            {view === 'evaluator' && (
                <EvaluatorRenderer onBack={() => setView('chat')} />
            )}
        </>
    );
};

export default MainApp;
