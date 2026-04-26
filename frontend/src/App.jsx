import React from 'react';
import ChatInterface from './components/ChatInterface';
import SkillManager from './components/SkillManager';
import LanguageSelector from './components/LanguageSelector';
import { MessageSquare, LogOut } from 'lucide-react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { LanguageProvider, useLanguage } from './contexts/LanguageContext';
import LoginPage from './pages/LoginPage';
import PostLoginLanding from './pages/PostLoginLanding';
import SettingsPage from './pages/SettingsPage';
import SetupWizard from './pages/SetupWizard';
import { Settings as SettingsIcon } from 'lucide-react';

function GlobalBackground({ children }) {
  const [brand, setBrand] = React.useState({ logo_url: '', background_url: '' });

  React.useEffect(() => {
    fetch('/api/brand').then(r => r.ok ? r.json() : {}).then(d => setBrand(b => ({ ...b, ...d }))).catch(() => {});
  }, []);

  const globalLogo = brand.logo_url ? (
    <div className="absolute top-4 left-4 sm:top-5 sm:left-6 z-[100] pointer-events-none">
      <img src={brand.logo_url} alt="Logo Institución" className="h-[150px] w-auto object-contain drop-shadow-md" />
    </div>
  ) : null;

  if (!brand.background_url) return (
    <div className="min-h-screen bg-gray-50 relative">
      {globalLogo}
      {children}
    </div>
  );
  
  return (
    <div 
      className="min-h-screen bg-cover bg-center bg-no-repeat bg-fixed relative"
      style={{ backgroundImage: `url(${brand.background_url})` }}
    >
      <div className="min-h-screen bg-gradient-to-r from-slate-900 to-slate-900/50 relative">
        {globalLogo}
        {children}
      </div>
    </div>
  );
}

function AppShell() {
  const { user, loading, logout, showPostLoginLanding, dismissPostLoginLanding } = useAuth();
  const { t } = useLanguage();
  const [showSettings, setShowSettings] = React.useState(false);
  const [setupRequired, setSetupRequired] = React.useState(null);

  React.useEffect(() => {
    fetch('/api/system/status').then(r => r.ok ? r.json() : { setup_required: false })
      .then(d => setSetupRequired(d.setup_required))
      .catch(() => setSetupRequired(false));
  }, []);

  if (loading || setupRequired === null) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-transparent">
        <p className="text-gray-600 font-medium">{t('loading')}</p>
      </div>
    );
  }

  if (setupRequired) {
    return <SetupWizard onComplete={() => setSetupRequired(false)} />;
  }

  if (!user) {
    return <LoginPage />;
  }

  // Animación del contenedor dinámico:
  // Si está en el landing page: max-w-3xl, h-auto
  // Si está en chat o settings: max-w-[70vw], h-[80vh]
  const isLanding = showPostLoginLanding && !showSettings;

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center justify-center px-2 py-10 relative">
      
      {/* User Controls Fijos en Esquina Superior Derecha */}
      <div className="absolute top-4 right-4 flex items-center gap-4 bg-white/80 p-2 pl-4 rounded-full shadow-sm backdrop-blur-sm z-50 transition-all duration-500">
        <div className="flex flex-col items-end pr-2 border-r border-gray-200 hidden sm:flex">
          <span className="text-sm font-semibold text-gray-800 leading-tight">
            {user?.name || user?.full_name || 'Usuario'}
            {user?.role === 'admin' && <span className="ml-1 text-[10px] text-blue-600 font-bold uppercase">(Admin)</span>}
          </span>
          <span className="text-xs text-gray-500 leading-tight">{user?.email}</span>
        </div>
        <LanguageSelector />
        
        {user?.role === 'admin' && (
          <button
            onClick={() => setShowSettings(!showSettings)}
            title="Configuraciones"
            className={`flex items-center gap-1 text-sm transition pr-2 ${showSettings ? 'text-blue-600' : 'text-gray-600 hover:text-blue-600'}`}
          >
            <SettingsIcon className="w-4 h-4" />
          </button>
        )}

        <button
          onClick={logout}
          title={t('logout')}
          className="flex items-center gap-1 text-sm text-gray-600 hover:text-red-600 transition pr-2"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>

      {/* Contenedor Blanco Dinámico (Canvas) */}
      <div 
        className={`bg-white border shadow-2xl rounded-3xl overflow-hidden flex flex-col transition-all duration-700 ease-in-out origin-center z-[110]
          ${isLanding 
            ? 'w-full max-w-3xl border-blue-100 min-h-[400px]' 
            : 'w-[95vw] md:w-[85vw] lg:w-[90vw] h-[90vh] border-gray-200'
          }`}
      >
        {showSettings ? (
          <SettingsPage onClose={() => setShowSettings(false)} />
        ) : isLanding ? (
          <PostLoginLanding user={user} onContinue={dismissPostLoginLanding} onConfig={() => setShowSettings(true)} />
        ) : (
          <div className="flex-1 flex flex-col h-full w-full opacity-0 animate-[fadeIn_0.5s_ease-in-out_0.3s_forwards]">
            <header className="bg-slate-50 border-b border-gray-200 p-4 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-6 h-6 text-blue-600" />
                <h1 className="text-xl font-bold text-gray-800">AsistIAG Orchestrator</h1>
              </div>
              <div className="flex items-center gap-4">
                {user?.role === 'admin' && <SkillManager />}
              </div>
            </header>

            <main className="flex-1 overflow-hidden relative">
              <ChatInterface />
            </main>
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  return (
    <LanguageProvider>
      <AuthProvider>
        <GlobalBackground>
          <AppShell />
        </GlobalBackground>
      </AuthProvider>
    </LanguageProvider>
  );
}

export default App;

