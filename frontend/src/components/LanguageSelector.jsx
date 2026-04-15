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

  return (
    <select
      value={lang}
      onChange={(e) => setLang(e.target.value)}
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
