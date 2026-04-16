import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import LanguageSelector from '../components/LanguageSelector';
import logoApp from '../assets/logo_app.png';

const GoogleIcon = () => (
  <svg width="18" height="18" viewBox="0 0 48 48">
    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
    <path fill="#FBBC05" d="M10.53 28.59a14.5 14.5 0 0 1 0-9.18l-7.98-6.19a24.01 24.01 0 0 0 0 21.56l7.98-6.19z"/>
    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
  </svg>
);

const MicrosoftIcon = () => (
  <svg width="18" height="18" viewBox="0 0 21 21">
    <rect x="1" y="1" width="9" height="9" fill="#F25022"/>
    <rect x="11" y="1" width="9" height="9" fill="#7FBA00"/>
    <rect x="1" y="11" width="9" height="9" fill="#00A4EF"/>
    <rect x="11" y="11" width="9" height="9" fill="#FFB900"/>
  </svg>
);

const PROVIDER_LABELS = {
  GOOGLE: { label: 'continue_google', Icon: GoogleIcon, href: '/auth/login/google' },
  MICROSOFT: { label: 'continue_microsoft', Icon: MicrosoftIcon, href: '/auth/login/microsoft' },
  OAUTH2: { label: 'continue_uchile', Icon: () => <span>🎓</span>, href: '/auth/login/uchile' },
  LOCAL: { label: null },  // rendered separately as a form
};

export default function LoginPage() {
  const { loginWithToken } = useAuth();
  const { t } = useLanguage();
  const [mode, setMode] = useState(null);   // null = loading
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch('/auth/mode')
      .then(r => r.json())
      .then(d => setMode(d.mode || ''))
      .catch(() => setMode(''));
  }, []);

  const showAll = mode === '';
  const showProvider = (p) => showAll || mode === p;

  async function handleLocalLogin(e) {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      const res = await fetch('/auth/login/local', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) {
        const d = await res.json();
        setError(t('auth_error'));
        return;
      }
      const { access_token } = await res.json();
      loginWithToken(access_token);
    } catch {
      setError(t('network_error'));
    } finally {
      setBusy(false);
    }
  }

  if (mode === null) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-transparent">
        <p className="text-gray-500 font-medium">{t('loading')}</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-transparent p-4">
      <div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-sm space-y-6">
        <div className="text-center">
          <img src={logoApp} alt="Logo Aplicación" className="h-14 w-auto object-contain mx-auto drop-shadow-sm mb-2" />
          <p className="text-sm text-gray-500 mt-1">{t('login_subtitle')}</p>
        </div>

        {/* OAuth providers */}
        <div className="space-y-3">
          {Object.entries(PROVIDER_LABELS).map(([key, cfg]) => {
            if (!showProvider(key) || key === 'LOCAL') return null;
            return (
              <a
                key={key}
                href={cfg.href}
                className="flex items-center justify-center gap-3 w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
              >
                <cfg.Icon />
                {t(cfg.label)}
              </a>
            );
          })}
        </div>

        {/* Divider */}
        {showAll && showProvider('LOCAL') && (
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <div className="flex-1 h-px bg-gray-200" />
            <span>{t('or_local')}</span>
            <div className="flex-1 h-px bg-gray-200" />
          </div>
        )}

        {/* Local login form */}
        {showProvider('LOCAL') && (
          <form onSubmit={handleLocalLogin} className="space-y-3">
            <input
              type="email"
              required
              placeholder={t('email')}
              value={email}
              onChange={e => setEmail(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="password"
              required
              placeholder={t('password')}
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {error && <p className="text-red-500 text-xs">{error}</p>}
            <button
              type="submit"
              disabled={busy}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2.5 text-sm font-medium transition disabled:opacity-50"
            >
              {busy ? t('entering') : t('login_button')}
            </button>
          </form>
        )}
      </div>
      
      {/* Selector de idioma bajo el formulario */}
      <div className="mt-8 flex justify-center bg-white shadow-sm p-2 rounded-full border border-gray-200">
        <LanguageSelector />
      </div>
    </div>
  );
}
