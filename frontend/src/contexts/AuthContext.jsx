import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AuthContext = createContext(null);

const TOKEN_KEY = 'rubricai_token';
const POST_LOGIN_LANDING_KEY = 'rubricai_show_post_login_landing';
const API_BASE = '';  // proxy via Vite → localhost:8000

/** Wrap native fetch to inject Bearer token for /api/* calls. */
function patchFetch(getToken) {
  const _orig = window._origFetch || window.fetch;
  window._origFetch = _orig;
  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input instanceof Request ? input.url : String(input);
    if (url.startsWith('/api/')) {
      const token = getToken();
      if (token) {
        init.headers = { ...(init.headers || {}), Authorization: `Bearer ${token}` };
      }
    }
    return _orig(input, init);
  };
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [loading, setLoading] = useState(true);
  const [showPostLoginLanding, setShowPostLoginLanding] = useState(() =>
    sessionStorage.getItem(POST_LOGIN_LANDING_KEY) === '1'
  );

  // Patch fetch once on mount
  useEffect(() => {
    patchFetch(() => localStorage.getItem(TOKEN_KEY));
  }, []);

  // On mount: check for ?token=... in URL (post-OAuth redirect)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get('token');
    if (urlToken) {
      localStorage.setItem(TOKEN_KEY, urlToken);
      sessionStorage.setItem(POST_LOGIN_LANDING_KEY, '1');
      setToken(urlToken);
      setShowPostLoginLanding(true);
      // Clean URL
      const clean = window.location.pathname;
      window.history.replaceState({}, '', clean);
    }
  }, []);

  // Validate token and load user info
  useEffect(() => {
    if (!token) { setLoading(false); return; }
    fetch('/auth/me', { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(u => setUser(u))
      .catch(() => { localStorage.removeItem(TOKEN_KEY); setToken(null); })
      .finally(() => setLoading(false));
  }, [token]);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(POST_LOGIN_LANDING_KEY);
    setToken(null);
    setUser(null);
    setShowPostLoginLanding(false);
  }, []);

  const loginWithToken = useCallback((t) => {
    localStorage.setItem(TOKEN_KEY, t);
    sessionStorage.setItem(POST_LOGIN_LANDING_KEY, '1');
    setToken(t);
    setShowPostLoginLanding(true);
  }, []);

  const dismissPostLoginLanding = useCallback(() => {
    sessionStorage.removeItem(POST_LOGIN_LANDING_KEY);
    setShowPostLoginLanding(false);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        logout,
        loginWithToken,
        showPostLoginLanding,
        dismissPostLoginLanding,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
