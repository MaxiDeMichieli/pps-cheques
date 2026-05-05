import { useEffect, useMemo, useState } from 'react';
import type { Cheque, ChequePatch } from '../api/types';
import { imagenUrl, patchCheque } from '../api/client';
import { ImageViewer } from './ImageViewer';

interface Props {
  cheque: Cheque;
  onClose: () => void;
  onSaved: (updated: Cheque) => void;
}

const FIELDS: Array<{ key: keyof Cheque; label: string; type?: string }> = [
  { key: 'monto', label: 'Monto', type: 'number' },
  { key: 'fecha_emision', label: 'Fecha emisión', type: 'date' },
  { key: 'fecha_pago', label: 'Fecha pago', type: 'date' },
  { key: 'sucursal', label: 'Sucursal' },
  { key: 'numero_sucursal', label: 'Nº Sucursal' },
  { key: 'numero_cheque', label: 'Nº Cheque' },
  { key: 'numero_cuenta', label: 'Nº Cuenta' },
  { key: 'cuit_librador', label: 'CUIT librador' },
  { key: 'nombre_librador', label: 'Nombre librador' },
];

export function ChequeDetail({ cheque, onClose, onSaved }: Props) {
  const [draft, setDraft] = useState<ChequePatch>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset draft cuando cambia el cheque
  useEffect(() => {
    setDraft({});
    setError(null);
  }, [cheque.id]);

  // Cierre con tecla Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const dirty = Object.keys(draft).length > 0;

  const value = (key: keyof Cheque): string => {
    if (key in draft) {
      const v = draft[key as keyof ChequePatch];
      return v == null ? '' : String(v);
    }
    const v = cheque[key];
    return v == null ? '' : String(v);
  };

  const setField = (key: keyof Cheque, raw: string, type?: string) => {
    const parsed: string | number | null =
      raw === '' ? null
      : type === 'number' ? Number(raw)
      : raw;
    setDraft((d) => ({ ...d, [key]: parsed }));
  };

  const fieldChanged = (key: keyof Cheque): boolean => key in draft;

  const confidenceFor = (key: keyof Cheque): number | null => {
    if (key === 'monto') return cheque.monto_llm_confidence;
    if (key === 'fecha_emision') return cheque.fecha_emision_llm_confidence;
    if (key === 'fecha_pago') return cheque.fecha_pago_llm_confidence;
    return null;
  };

  const inputBorder = (key: keyof Cheque): string => {
    if (fieldChanged(key)) return 'border-orange-400 ring-1 ring-orange-300';
    if (cheque[key] === null || cheque[key] === '') return 'border-red-300';
    const conf = confidenceFor(key);
    if (conf !== null && conf < 0.7) return 'border-yellow-400';
    return 'border-gray-300 dark:border-gray-700';
  };

  const onSave = async () => {
    if (!dirty) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await patchCheque(cheque.id, draft);
      onSaved(updated);
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const onCancel = () => {
    if (dirty && !window.confirm('Hay cambios sin guardar. ¿Descartar?')) return;
    onClose();
  };

  const adjacentInfo = useMemo(
    () => `Página ${cheque.pagina} · Cheque ${cheque.indice_en_pagina}`,
    [cheque.pagina, cheque.indice_en_pagina],
  );

  return (
    <div
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={onCancel}
    >
      <div
        className="bg-white dark:bg-gray-900 rounded-lg shadow-xl max-w-7xl w-full max-h-[95vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <h2 className="font-semibold">{adjacentInfo}</h2>
          <button
            type="button"
            onClick={onCancel}
            className="text-gray-500 hover:text-gray-900 dark:hover:text-gray-100 text-2xl leading-none px-2"
            aria-label="Cerrar"
          >
            ×
          </button>
        </div>

        {/* Body: imagen + form */}
        <div className="flex-1 overflow-hidden grid grid-cols-1 lg:grid-cols-[1fr_420px]">
          {/* Imagen */}
          <ImageViewer
            src={imagenUrl(cheque.id)}
            alt={adjacentInfo}
            resetKey={cheque.id}
          />

          {/* Form */}
          <div className="p-4 overflow-y-auto border-l border-gray-200 dark:border-gray-700">
            <div className="grid grid-cols-1 gap-3">
              {FIELDS.map((f) => (
                <label key={f.key} className="flex flex-col gap-1">
                  <span className="text-xs font-medium text-gray-500 flex items-center gap-2">
                    {f.label}
                    {fieldChanged(f.key) && (
                      <span className="text-orange-500">● modificado</span>
                    )}
                    {confidenceFor(f.key) !== null && (
                      <span className="text-gray-400">
                        conf {(confidenceFor(f.key)! * 100).toFixed(0)}%
                      </span>
                    )}
                  </span>
                  <input
                    type={f.type ?? 'text'}
                    step={f.type === 'number' ? 'any' : undefined}
                    value={value(f.key)}
                    onChange={(e) => setField(f.key, e.target.value, f.type)}
                    disabled={!cheque.extracted}
                    className={`bg-white dark:bg-gray-900 border rounded px-2 py-1.5 text-sm
                      ${inputBorder(f.key)} disabled:opacity-50 focus:outline-none focus:ring-1 focus:ring-blue-500`}
                  />
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between gap-3">
          <div className="text-xs">
            {error && <span className="text-red-600">{error}</span>}
            {!error && dirty && (
              <span className="text-orange-500">Cambios sin guardar</span>
            )}
            {!error && !dirty && (
              <span className="text-gray-400">Sin cambios</span>
            )}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-1.5 text-sm rounded border border-gray-300 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              Cancelar
            </button>
            <button
              type="button"
              onClick={onSave}
              disabled={!dirty || saving}
              className="px-4 py-1.5 text-sm rounded bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {saving ? 'Guardando...' : 'Guardar'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
