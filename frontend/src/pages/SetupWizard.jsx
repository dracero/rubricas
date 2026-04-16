import React, { useState } from 'react';
import { Sparkles, Upload, Globe, UserPlus, CheckCircle } from 'lucide-react';

const LANGUAGES = [
  { code: 'es', label: 'Español' },
  { code: 'en', label: 'English' },
  { code: 'pt', label: 'Português' },
  { code: 'gl', label: 'Galego' },
];

export default function SetupWizard({ onComplete }) {
  const [form, setForm] = useState({
    institucion_nombre: '',
    admin_email: '',
    admin_name: '',
    admin_password: '',
    default_language: 'es',
  });
  const [logo, setLogo] = useState(null);
  const [background, setBackground] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    const fd = new FormData();
    fd.append('institucion_nombre', form.institucion_nombre);
    fd.append('admin_email', form.admin_email);
    fd.append('admin_name', form.admin_name);
    fd.append('admin_password', form.admin_password);
    fd.append('default_language', form.default_language);
    if (logo) fd.append('logo', logo);
    if (background) fd.append('background', background);

    try {
      const res = await fetch('/api/system/setup', { method: 'POST', body: fd });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Error en la configuración inicial.');
      }
      onComplete();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex items-center justify-center p-4">
      <form
        onSubmit={handleSubmit}
        className="bg-white rounded-3xl shadow-2xl w-full max-w-2xl p-8 md:p-10 space-y-6 animate-[fadeIn_0.5s_ease-in-out]"
      >
        <div className="flex items-center gap-3">
          <div className="bg-blue-100 text-blue-600 p-2 rounded-xl">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-800">Configuración Inicial</h1>
            <p className="text-sm text-slate-500">Configura tu instancia de RubricAI por primera vez.</p>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 text-red-700 p-3 rounded-xl text-sm border border-red-200">{error}</div>
        )}

        {/* Institución */}
        <fieldset className="space-y-3">
          <legend className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Globe className="w-4 h-4" /> Institución
          </legend>
          <input
            required
            placeholder="Nombre de la institución"
            value={form.institucion_nombre}
            onChange={set('institucion_nombre')}
            className="w-full border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
          />
          <select
            value={form.default_language}
            onChange={set('default_language')}
            className="w-full border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </fieldset>

        {/* Admin */}
        <fieldset className="space-y-3">
          <legend className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <UserPlus className="w-4 h-4" /> Cuenta Administrador
          </legend>
          <input
            required type="text"
            placeholder="Nombre completo"
            value={form.admin_name}
            onChange={set('admin_name')}
            className="w-full border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
          />
          <input
            required type="email"
            placeholder="Correo electrónico"
            value={form.admin_email}
            onChange={set('admin_email')}
            className="w-full border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
          />
          <input
            required type="password" minLength={6}
            placeholder="Contraseña (mín. 6 caracteres)"
            value={form.admin_password}
            onChange={set('admin_password')}
            className="w-full border rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
          />
        </fieldset>

        {/* Brand files */}
        <fieldset className="space-y-3">
          <legend className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Upload className="w-4 h-4" /> Imagen institucional (opcional)
          </legend>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <label className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-4 cursor-pointer transition text-center ${logo ? 'border-green-400 bg-green-50' : 'hover:border-blue-400'}`}>
              {logo ? <CheckCircle className="w-5 h-5 text-green-500 mb-1" /> : <Upload className="w-5 h-5 text-slate-400 mb-1" />}
              <span className="text-xs text-slate-500 mb-1">Logo</span>
              <span className={`text-[10px] truncate max-w-full ${logo ? 'text-green-600 font-medium' : 'text-slate-400'}`}>{logo ? logo.name : 'PNG / JPG'}</span>
              <input type="file" accept="image/*" className="hidden" onChange={(e) => setLogo(e.target.files[0])} />
            </label>
            <label className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-4 cursor-pointer transition text-center ${background ? 'border-green-400 bg-green-50' : 'hover:border-blue-400'}`}>
              {background ? <CheckCircle className="w-5 h-5 text-green-500 mb-1" /> : <Upload className="w-5 h-5 text-slate-400 mb-1" />}
              <span className="text-xs text-slate-500 mb-1">Fondo</span>
              <span className={`text-[10px] truncate max-w-full ${background ? 'text-green-600 font-medium' : 'text-slate-400'}`}>{background ? background.name : 'PNG / JPG'}</span>
              <input type="file" accept="image/*" className="hidden" onChange={(e) => setBackground(e.target.files[0])} />
            </label>
          </div>
        </fieldset>

        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 rounded-xl transition"
        >
          {submitting ? 'Configurando...' : 'Completar Configuración'}
        </button>
      </form>
    </div>
  );
}
