import React from 'react';
import { Sparkles, ArrowRight, ShieldCheck, FileText, Settings, BookOpen, PenLine } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export default function PostLoginLanding({ user, onContinue, onConfig }) {
  const { t } = useLanguage();
  const role = user?.role;

  // Define features per role
  // rubricador: generate + repository + writing assistant
  // verificador: all of the above + evaluation
  // admin: everything
  const features = [
    {
      icon: <FileText className="w-5 h-5 text-blue-600" />,
      title: t('feature_1_title'),
      desc: t('feature_1_desc'),
      roles: ['admin', 'verificador', 'rubricador'],
    },
    {
      icon: <ShieldCheck className="w-5 h-5 text-emerald-600" />,
      title: t('feature_2_title'),
      desc: t('feature_2_desc'),
      roles: ['admin', 'verificador'],
    },
    {
      icon: <Sparkles className="w-5 h-5 text-violet-600" />,
      title: t('feature_3_title'),
      desc: t('feature_3_desc'),
      roles: ['admin', 'verificador', 'rubricador'],
    },
  ].filter(f => f.roles.includes(role));

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

        {features.length > 0 && (
          <div className={`mt-6 grid grid-cols-1 gap-3 ${features.length > 1 ? 'md:grid-cols-' + features.length : ''}`}>
            {features.map((f, i) => (
              <div key={i} className="rounded-xl border border-slate-200 p-4 bg-slate-50">
                {f.icon}
                <p className="mt-2 text-sm font-semibold text-slate-800">{f.title}</p>
                <p className="mt-1 text-xs text-slate-600">{f.desc}</p>
              </div>
            ))}
          </div>
        )}

        <div className="mt-8 flex justify-between items-center">
          {role === 'admin' ? (
            <button
              type="button"
              onClick={onConfig}
              className="inline-flex items-center gap-2 text-slate-500 hover:text-blue-600 font-medium px-4 py-2.5 rounded-xl transition hover:bg-blue-50"
            >
              <Settings className="w-5 h-5" />
              <span>Configuración</span>
            </button>
          ) : (
            <div />
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
