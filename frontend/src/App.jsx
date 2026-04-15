import React from 'react';
import ChatInterface from './components/ChatInterface';
import SkillManager from './components/SkillManager';
import LanguageSelector from './components/LanguageSelector';
import { LanguageProvider, useLanguage } from './contexts/LanguageContext';
import { MessageSquare } from 'lucide-react';

function AppContent() {
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow p-4 flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-6 h-6 text-blue-600" />
          <h1 className="text-xl font-bold text-gray-800">{t('header.title')}</h1>
        </div>
        <div className="flex items-center gap-3">
          <LanguageSelector />
          <SkillManager />
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 flex flex-col">
        <ChatInterface />
      </main>
    </div>
  );
}

function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

export default App;

