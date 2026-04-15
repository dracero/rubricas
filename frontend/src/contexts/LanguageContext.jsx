import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { translations } from '../i18n/translations';

const SUPPORTED_LANGUAGES = ['es', 'gl', 'en', 'pt'];
const DEFAULT_LANGUAGE = 'es';
const STORAGE_KEY = 'rubricai_lang';

function readStoredLang() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && SUPPORTED_LANGUAGES.includes(stored)) {
      return stored;
    }
    // Invalid or missing — reset to default
    localStorage.setItem(STORAGE_KEY, DEFAULT_LANGUAGE);
  } catch {
    // localStorage unavailable (private browsing, etc.)
  }
  return DEFAULT_LANGUAGE;
}

const LanguageContext = createContext({
  lang: DEFAULT_LANGUAGE,
  setLang: () => {},
  t: (key, fallback) => fallback || key,
});

export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(readStoredLang);

  const setLang = useCallback((code) => {
    const validCode = SUPPORTED_LANGUAGES.includes(code) ? code : DEFAULT_LANGUAGE;
    try {
      localStorage.setItem(STORAGE_KEY, validCode);
    } catch {
      // localStorage unavailable
    }
    setLangState(validCode);
  }, []);

  const t = useCallback((key, fallback) => {
    const currentCatalog = translations[lang];
    if (currentCatalog && currentCatalog[key]) {
      return currentCatalog[key];
    }
    // Fallback to es
    const esCatalog = translations[DEFAULT_LANGUAGE];
    if (esCatalog && esCatalog[key]) {
      return esCatalog[key];
    }
    // Last resort: fallback param or key itself
    return fallback || key;
  }, [lang]);

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}

export { SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, STORAGE_KEY };
export default LanguageContext;
