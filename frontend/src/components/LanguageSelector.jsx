import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';

const LANGUAGES = [
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'gl', name: 'Galego', flag: '🇬🇱' },
  { code: 'en', name: 'English', flag: '🇬🇧' },
  { code: 'pt', name: 'Português', flag: '🇵🇹' },
];

const LanguageSelector = () => {
  const { lang, setLang } = useLanguage();

  const handleChange = async (e) => {
    const newLang = e.target.value;
    setLang(newLang);
    // Reset backend chat session so the agent responds in the new language
    try {
      await fetch('/api/chat/reset', { method: 'POST' });
    } catch { /* ignore */ }
  };

  return (
    <select
      value={lang}
      onChange={handleChange}
      className="text-sm bg-gray-50 border border-gray-200 rounded-lg px-2 py-1.5 text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer transition"
      aria-label="Language"
    >
      {LANGUAGES.map((l) => (
        <option key={l.code} value={l.code}>
          {l.flag} {l.name}
        </option>
      ))}
    </select>
  );
};

export default LanguageSelector;
