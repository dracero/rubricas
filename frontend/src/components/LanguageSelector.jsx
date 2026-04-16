import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Globe } from 'lucide-react';

export default function LanguageSelector() {
  const { lang, setLang, t } = useLanguage();

  return (
    <div className="flex items-center gap-2 text-sm text-gray-600">
      <Globe className="w-4 h-4" />
      <select
        value={lang}
        onChange={(e) => setLang(e.target.value)}
        className="bg-transparent border-none outline-none cursor-pointer focus:ring-0 text-gray-700 font-medium"
      >
        <option value="es">{t('lang_es')}</option>
        <option value="gl">{t('lang_gl')}</option>
        <option value="pt">{t('lang_pt')}</option>
        <option value="en">{t('lang_en')}</option>
      </select>
    </div>
  );
}