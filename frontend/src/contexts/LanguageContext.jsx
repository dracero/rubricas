import React, { createContext, useContext, useState, useEffect } from 'react';
import translations from '../locales/translations.json';

const LanguageContext = createContext();

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState(() => {
    return localStorage.getItem('rubricai_lang') || 'es';
  });

  useEffect(() => {
    // If no saved preference, fetch default from backend
    if (!localStorage.getItem('rubricai_lang')) {
      fetch('/api/brand').then(r => r.ok ? r.json() : {})
        .then(d => { if (d.default_language) setLang(d.default_language); })
        .catch(() => {});
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('rubricai_lang', lang);
  }, [lang]);

  const t = (key) => {
    return translations[lang]?.[key] || translations['es'][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}