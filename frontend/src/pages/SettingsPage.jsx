import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import { Settings, ShieldAlert, ArrowLeft, Loader2, RefreshCw, Save, Upload, ExternalLink, UserPlus, Trash2, Pencil, X, Check } from 'lucide-react';

const LANGUAGE_OPTIONS = [
  { code: 'es', label: 'Español (neutro)' },
  { code: 'pt', label: 'Português' },
  { code: 'gl', label: 'Galego' },
  { code: 'en', label: 'English' },
];

const AUTH_MODE_OPTIONS = [
  { code: 'local', label: 'Local (email + contraseña)' },
  { code: 'google', label: 'Google OAuth' },
  { code: 'both', label: 'Local + Google OAuth' },
];

const ROLE_LABELS = {
  admin:       { label: 'Admin',       color: 'bg-purple-100 text-purple-800 border-purple-200' },
  verificador: { label: 'Verificador', color: 'bg-blue-100 text-blue-800 border-blue-200' },
  rubricador:  { label: 'Rubricador',  color: 'bg-green-100 text-green-800 border-green-200' },
};

const ROLE_DESCRIPTIONS = {
  admin:       'Acceso total: configuración, skills, usuarios y todas las funciones.',
  verificador: 'Puede usar todos los skills y evaluar/verificar documentos.',
  rubricador:  'Puede generar rúbricas, usar el repositorio y el asistente de redacción. Sin acceso a evaluación.',
};

function RoleBadge({ role }) {
  const r = ROLE_LABELS[role] || { label: role, color: 'bg-gray-100 text-gray-700 border-gray-200' };
  return (
    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${r.color}`}>
      {r.label}
    </span>
  );
}

function UserManagement({ authHeaders, currentUserEmail }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showCreate, setShowCreate] = useState(false);
  const [editingEmail, setEditingEmail] = useState(null);
  const [editData, setEditData] = useState({});
  const [newUser, setNewUser] = useState({ email: '', name: '', password: '', role: 'rubricador' });
  const [saving, setSaving] = useState(false);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/users', { headers: authHeaders });
      if (!res.ok) throw new Error('No se pudo obtener la lista de usuarios.');
      setUsers(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchUsers(); }, []);

  const handleCreate = async () => {
    if (!newUser.email || !newUser.password || !newUser.name) {
      setError('Email, nombre y contraseña son requeridos.'); return;
    }
    setSaving(true); setError(null); setSuccess(null);
    try {
      const res = await fetch('/api/users', {
        method: 'POST',
        headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(newUser),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Error al crear usuario.'); }
      setSuccess(`Usuario ${newUser.email} creado.`);
      setNewUser({ email: '', name: '', password: '', role: 'rubricador' });
      setShowCreate(false);
      fetchUsers();
    } catch (e) { setError(e.message); }
    finally { setSaving(false); }
  };

  const handleUpdate = async (email) => {
    setSaving(true); setError(null); setSuccess(null);
    try {
      const res = await fetch(`/api/users/${encodeURIComponent(email)}`, {
        method: 'PUT',
        headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(editData),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Error al actualizar.'); }
      setSuccess(`Usuario ${email} actualizado.`);
      setEditingEmail(null);
      fetchUsers();
    } catch (e) { setError(e.message); }
    finally { setSaving(false); }
  };

  const handleDelete = async (email) => {
    if (!confirm(`¿Eliminar al usuario ${email}?`)) return;
    setError(null); setSuccess(null);
    try {
      const res = await fetch(`/api/users/${encodeURIComponent(email)}`, {
        method: 'DELETE', headers: authHeaders,
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Error al eliminar.'); }
      setSuccess(`Usuario ${email} eliminado.`);
      fetchUsers();
    } catch (e) { setError(e.message); }
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
      <div className="flex items-center justify-between mb-4 border-b border-gray-100 pb-4">
        <div>
          <h2 className="text-lg font-bold text-gray-800">Gestión de Usuarios</h2>
          <p className="text-sm text-gray-500">Creá, editá o eliminá usuarios y asigná roles.</p>
        </div>
        <button
          onClick={() => { setShowCreate(!showCreate); setError(null); }}
          className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-4 py-2 rounded-xl transition"
        >
          <UserPlus className="w-4 h-4" /> Nuevo usuario
        </button>
      </div>

      {error && <p className="text-red-600 text-sm mb-3 bg-red-50 p-3 rounded-xl border border-red-100">{error}</p>}
      {success && <p className="text-green-700 text-sm mb-3 bg-green-50 p-3 rounded-xl border border-green-100">{success}</p>}

      {/* Descripción de roles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        {Object.entries(ROLE_DESCRIPTIONS).map(([role, desc]) => (
          <div key={role} className="bg-slate-50 border border-slate-100 rounded-xl p-3">
            <RoleBadge role={role} />
            <p className="text-xs text-slate-500 mt-1.5">{desc}</p>
          </div>
        ))}
      </div>

      {/* Formulario nuevo usuario */}
      {showCreate && (
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 mb-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <input placeholder="Email" className="border rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-300"
            value={newUser.email} onChange={e => setNewUser({ ...newUser, email: e.target.value })} />
          <input placeholder="Nombre" className="border rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-300"
            value={newUser.name} onChange={e => setNewUser({ ...newUser, name: e.target.value })} />
          <input placeholder="Contraseña" type="password" className="border rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-300"
            value={newUser.password} onChange={e => setNewUser({ ...newUser, password: e.target.value })} />
          <select className="border rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-300 bg-white"
            value={newUser.role} onChange={e => setNewUser({ ...newUser, role: e.target.value })}>
            <option value="rubricador">Rubricador</option>
            <option value="verificador">Verificador</option>
            <option value="admin">Admin</option>
          </select>
          <div className="md:col-span-2 flex gap-2">
            <button onClick={handleCreate} disabled={saving}
              className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-semibold px-4 py-2 rounded-xl transition">
              <Check className="w-4 h-4" /> {saving ? 'Creando...' : 'Crear usuario'}
            </button>
            <button onClick={() => setShowCreate(false)}
              className="flex items-center gap-1.5 text-gray-600 hover:bg-gray-100 text-sm px-4 py-2 rounded-xl transition">
              <X className="w-4 h-4" /> Cancelar
            </button>
          </div>
        </div>
      )}

      {/* Lista de usuarios */}
      {loading ? (
        <div className="flex justify-center py-8"><Loader2 className="w-6 h-6 animate-spin text-blue-400" /></div>
      ) : (
        <div className="space-y-2">
          {users.map(u => (
            <div key={u.email} className="flex items-center gap-3 p-3 rounded-xl border border-gray-100 hover:bg-slate-50 transition">
              {editingEmail === u.email ? (
                <>
                  <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-2">
                    <input placeholder="Nombre" className="border rounded-lg px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-blue-300"
                      defaultValue={u.name} onChange={e => setEditData({ ...editData, name: e.target.value })} />
                    <select className="border rounded-lg px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-blue-300 bg-white"
                      defaultValue={u.role} onChange={e => setEditData({ ...editData, role: e.target.value })}>
                      <option value="rubricador">Rubricador</option>
                      <option value="verificador">Verificador</option>
                      <option value="admin">Admin</option>
                    </select>
                    <input placeholder="Nueva contraseña (opcional)" type="password"
                      className="border rounded-lg px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-blue-300"
                      onChange={e => setEditData({ ...editData, password: e.target.value || undefined })} />
                  </div>
                  <button onClick={() => handleUpdate(u.email)} disabled={saving}
                    className="p-1.5 text-green-600 hover:bg-green-50 rounded-lg transition" title="Guardar">
                    <Check className="w-4 h-4" />
                  </button>
                  <button onClick={() => setEditingEmail(null)}
                    className="p-1.5 text-gray-400 hover:bg-gray-100 rounded-lg transition" title="Cancelar">
                    <X className="w-4 h-4" />
                  </button>
                </>
              ) : (
                <>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-semibold text-gray-800 truncate">{u.name}</span>
                      <RoleBadge role={u.role} />
                      {!u.is_active && <span className="text-[10px] bg-red-100 text-red-700 border border-red-200 px-2 py-0.5 rounded-full font-bold">Inactivo</span>}
                      {u.email === currentUserEmail && <span className="text-[10px] text-gray-400">(vos)</span>}
                    </div>
                    <span className="text-xs text-gray-400">{u.email}</span>
                  </div>
                  <button onClick={() => { setEditingEmail(u.email); setEditData({}); }}
                    className="p-1.5 text-blue-500 hover:bg-blue-50 rounded-lg transition" title="Editar">
                    <Pencil className="w-4 h-4" />
                  </button>
                  {u.email !== currentUserEmail && (
                    <button onClick={() => handleDelete(u.email)}
                      className="p-1.5 text-red-400 hover:bg-red-50 rounded-lg transition" title="Eliminar">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SettingsPage({ onClose }) {
  const { t } = useLanguage();
  const { user } = useAuth();
  const [config, setConfig] = useState(null);
  const [editableKeys, setEditableKeys] = useState([]);
  const [edits, setEdits] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [logoFile, setLogoFile] = useState(null);
  const [bgFile, setBgFile] = useState(null);

  const token = localStorage.getItem('rubricai_token');
  const authHeaders = { 'Authorization': `Bearer ${token}` };

  const fetchConfig = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const res = await fetch('/api/config', { headers: authHeaders });
      if (!res.ok) {
        if (res.status === 403) throw new Error("No tienes permisos de administrador.");
        throw new Error("No se pudo obtener la configuración.");
      }
      const data = await res.json();
      setConfig(data.config);
      setEditableKeys(data.editable_keys || []);
      // Init edits from editable keys
      const initial = {};
      (data.editable_keys || []).forEach(k => { initial[k] = data.config[k] || ''; });
      setEdits(initial);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === 'admin') fetchConfig();
    else { setError("Vista restringida solo para perfil de Administrador."); setLoading(false); }
  }, [user]);

  const handleSaveSettings = async () => {
    setSaving(true); setError(null); setSuccess(null);
    try {
      const res = await fetch('/api/config/settings', {
        method: 'PUT',
        headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(edits),
      });
      if (!res.ok) throw new Error('Error al guardar configuración.');
      setSuccess('Configuración guardada.');
      fetchConfig();
    } catch (err) { setError(err.message); }
    finally { setSaving(false); }
  };

  const handleUploadBrand = async () => {
    if (!logoFile && !bgFile) return;
    setSaving(true); setError(null); setSuccess(null);
    try {
      const fd = new FormData();
      if (logoFile) fd.append('logo', logoFile);
      if (bgFile) fd.append('background', bgFile);
      const res = await fetch('/api/config/brand', { method: 'POST', headers: authHeaders, body: fd });
      if (!res.ok) throw new Error('Error al subir archivos.');
      setSuccess('Imágenes actualizadas.');
      setLogoFile(null); setBgFile(null);
      fetchConfig();
    } catch (err) { setError(err.message); }
    finally { setSaving(false); }
  };

  if (loading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center min-h-[400px] animate-pulse">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-4" />
        <p className="text-gray-500">Cargando parámetros del sistema...</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-50 relative animate-[fadeIn_0.3s_ease-in-out]">
      <header className="bg-white border-b border-gray-200 p-4 flex items-center justify-between shrink-0 sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <button 
            onClick={onClose}
            className="p-2 text-gray-500 hover:bg-gray-100 hover:text-blue-600 rounded-full transition"
            title="Volver"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <Settings className="w-6 h-6 text-slate-700" />
            <h1 className="text-xl font-bold text-gray-800">Panel de Administración</h1>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {user?.role === 'admin' && (
            <span className="bg-emerald-100 flex items-center gap-1.5 text-emerald-800 text-xs font-semibold px-3 py-1 rounded-full border border-emerald-200">
              <ShieldAlert className="w-3.5 h-3.5" />
              Modo Admin
            </span>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4 md:p-8">
        <div className="max-w-4xl mx-auto space-y-6">

          {/* Messages */}
          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-xl flex gap-3 items-start border border-red-100">
              <ShieldAlert className="w-5 h-5 mt-0.5 shrink-0" />
              <p className="font-medium text-sm">{error}</p>
            </div>
          )}
          {success && (
            <div className="bg-green-50 text-green-700 p-4 rounded-xl text-sm border border-green-200">{success}</div>
          )}

          {/* User Management */}
          {config && (
            <UserManagement authHeaders={authHeaders} currentUserEmail={user?.email} />
          )}

          {config && editableKeys.length > 0 && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
              <div className="flex items-center justify-between mb-6 border-b border-gray-100 pb-4">
                <div>
                  <h2 className="text-lg font-bold text-gray-800">Configuración Editable</h2>
                  <p className="text-sm text-gray-500">Parámetros que puedes modificar en tiempo real.</p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {editableKeys.map((key) => (
                  <div key={key}>
                    <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{key}</label>
                    {key === 'DEFAULT_LANGUAGE' ? (
                      <select
                        className="w-full border rounded-xl px-4 py-2 text-sm focus:ring-2 focus:ring-blue-300 outline-none bg-white"
                        value={edits[key] || 'es'}
                        onChange={(e) => setEdits({ ...edits, [key]: e.target.value })}
                      >
                        {LANGUAGE_OPTIONS.map((l) => (
                          <option key={l.code} value={l.code}>{l.label}</option>
                        ))}
                      </select>
                    ) : key === 'AUTH_MODE' ? (
                      <select
                        className="w-full border rounded-xl px-4 py-2 text-sm focus:ring-2 focus:ring-blue-300 outline-none bg-white"
                        value={edits[key] || 'local'}
                        onChange={(e) => setEdits({ ...edits, [key]: e.target.value })}
                      >
                        {AUTH_MODE_OPTIONS.map((a) => (
                          <option key={a.code} value={a.code}>{a.label}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        className="w-full border rounded-xl px-4 py-2 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
                        value={edits[key] || ''}
                        onChange={(e) => setEdits({ ...edits, [key]: e.target.value })}
                      />
                    )}
                  </div>
                ))}
              </div>
              <button
                onClick={handleSaveSettings}
                disabled={saving}
                className="mt-4 flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-semibold px-5 py-2 rounded-xl transition"
              >
                <Save className="w-4 h-4" /> {saving ? 'Guardando...' : 'Guardar Cambios'}
              </button>
            </div>
          )}

          {/* Brand Upload */}
          {config && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
              <div className="mb-4 border-b border-gray-100 pb-4">
                <h2 className="text-lg font-bold text-gray-800">Imagen Institucional</h2>
                <p className="text-sm text-gray-500">Sube o reemplaza el logo y fondo de la aplicación.</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="flex flex-col items-center gap-2">
                  <label className="flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-4 cursor-pointer hover:border-blue-400 transition text-center w-full">
                    <Upload className="w-5 h-5 text-slate-400 mb-1" />
                    <span className="text-xs text-slate-500">Logo</span>
                    <span className="text-[10px] text-slate-400">{logoFile ? logoFile.name : config.BRAND_LOGO_URL || 'Sin archivo'}</span>
                    <input type="file" accept="image/*" className="hidden" onChange={(e) => setLogoFile(e.target.files[0])} />
                  </label>
                  {config.BRAND_LOGO_URL && (
                    <a href={config.BRAND_LOGO_URL} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-xs text-blue-600 hover:underline">
                      <ExternalLink className="w-3 h-3" /> Ver logo actual
                    </a>
                  )}
                </div>
                <div className="flex flex-col items-center gap-2">
                  <label className="flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-4 cursor-pointer hover:border-blue-400 transition text-center w-full">
                    <Upload className="w-5 h-5 text-slate-400 mb-1" />
                    <span className="text-xs text-slate-500">Fondo</span>
                    <span className="text-[10px] text-slate-400">{bgFile ? bgFile.name : config.BRAND_BACKGROUND_URL || 'Sin archivo'}</span>
                    <input type="file" accept="image/*" className="hidden" onChange={(e) => setBgFile(e.target.files[0])} />
                  </label>
                  {config.BRAND_BACKGROUND_URL && (
                    <a href={config.BRAND_BACKGROUND_URL} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-xs text-blue-600 hover:underline">
                      <ExternalLink className="w-3 h-3" /> Ver fondo actual
                    </a>
                  )}
                </div>
              </div>
              {(logoFile || bgFile) && (
                <button
                  onClick={handleUploadBrand}
                  disabled={saving}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-semibold px-5 py-2 rounded-xl transition"
                >
                  <Upload className="w-4 h-4" /> {saving ? 'Subiendo...' : 'Subir Imágenes'}
                </button>
              )}
            </div>
          )}

          {/* Read-only env keys */}
          {config && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
              <div className="flex items-center justify-between mb-6 border-b border-gray-100 pb-4">
                <div>
                  <h2 className="text-lg font-bold text-gray-800">Parámetros del Entorno (solo lectura)</h2>
                  <p className="text-sm text-gray-500">Variables definidas en .env — no editables desde aquí.</p>
                </div>
                <button
                  onClick={fetchConfig}
                  className="flex items-center gap-2 text-sm text-blue-600 font-medium hover:bg-blue-50 px-3 py-1.5 rounded-lg transition"
                >
                  <RefreshCw className="w-4 h-4" /> Refrescar
                </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(config)
                  .filter(([key]) => !editableKeys.includes(key) && !key.startsWith('BRAND_'))
                  .map(([key, val]) => (
                    <div key={key} className="bg-slate-50 border border-slate-100 rounded-xl p-3">
                      <span className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{key}</span>
                      <span className={`text-sm font-medium break-all select-all ${val === '********' ? 'text-gray-400 italic' : val ? 'text-slate-700' : 'text-gray-300'}`}>
                        {val || '(Vacío / No definido)'}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}