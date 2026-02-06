import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import { MessageSquare } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow p-4 flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-6 h-6 text-blue-600" />
          <h1 className="text-xl font-bold text-gray-800">RubricAI Orchestrator</h1>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 flex flex-col">
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;
