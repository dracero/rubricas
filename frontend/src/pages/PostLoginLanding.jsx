import React from 'react';
import { Sparkles, ArrowRight, ShieldCheck, FileText, LogOut, Settings } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { useAuth } from '../contexts/AuthContext';
import LanguageSelector from '../components/LanguageSelector';

export default function PostLoginLanding({ user, onContinue, onConfig }) {
  const { t } = useLanguage();

  return (
    <div className="flex flex-col h-full items-center justify-center p-8 md:p-10 fade-in-up">
      <div className="w-full">
        <div className="inline-flex items-center gap-2 rounded-full bg-blue-100 text-blue-700 px-3 py-1 text-xs font-semibold">
          <Sparkles className="w-4 h-4" />
          {t('login_success')}
        </div>

        <h1 className="mt-4 text-3xl md:text-4xl font-bold text-slate-800 leading-tight">
          {t('welcome')}
        </h1>
        <p className="mt-2 text-slate-600 text-sm md:text-base">
          {t('logged_in_as')} <span className="font-semibold">{user?.email}</span>. {t('landing_desc')}
        </p>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="rounded-xl border border-slate-200 p-4 bg-slate-50">
            <FileText className="w-5 h-5 text-blue-600" />
            <p className="mt-2 text-sm font-semibold text-slate-800">{t('feature_1_title')}</p>
            <p className="mt-1 text-xs text-slate-600">{t('feature_1_desc')}</p>
          </div>
          <div className="rounded-xl border border-slate-200 p-4 bg-slate-50">
            <ShieldCheck className="w-5 h-5 text-emerald-600" />
            <p className="mt-2 text-sm font-semibold text-slate-800">{t('feature_2_title')}</p>
            <p className="mt-1 text-xs text-slate-600">{t('feature_2_desc')}</p>
          </div>
          <div className="rounded-xl border border-slate-200 p-4 bg-slate-50">
            <Sparkles className="w-5 h-5 text-violet-600" />
            <p className="mt-2 text-sm font-semibold text-slate-800">{t('feature_3_title')}</p>
            <p className="mt-1 text-xs text-slate-600">{t('feature_3_desc')}</p>
          </div>
        </div>

        <div className="mt-8 flex justify-between items-center">
          {user?.role === 'admin' ? (
            <button
              type="button"
              onClick={onConfig}
              className="inline-flex items-center gap-2 text-slate-500 hover:text-blue-600 font-medium px-4 py-2.5 rounded-xl transition hover:bg-blue-50"
            >
              <Settings className="w-5 h-5" />
              <span>Configuración</span>
            </button>
          ) : (
            <div></div> /* Espaciador si no es admin */
          )}

          <button
            type="button"
            onClick={onContinue}
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2.5 rounded-xl transition hover:scale-105 active:scale-95"
          >
            {t('continue_dashboard')}
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
