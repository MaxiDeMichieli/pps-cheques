import type { Cheque } from '../api/types';
import { imagenUrl } from '../api/client';

interface Props {
  cheque: Cheque;
  onOpen: (cheque: Cheque) => void;
}

function formatMonto(monto: number | null): string {
  if (monto == null) return '—';
  return new Intl.NumberFormat('es-AR', {
    style: 'currency',
    currency: 'ARS',
    maximumFractionDigits: 2,
  }).format(monto);
}

function formatFecha(fecha: string | null): string {
  if (!fecha) return '—';
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(fecha);
  if (!m) return fecha;
  return `${m[3]}/${m[2]}/${m[1]}`;
}

export function ChequeCard({ cheque, onOpen }: Props) {
  const missingCount = [
    cheque.monto, cheque.fecha_emision, cheque.fecha_pago,
    cheque.sucursal, cheque.numero_sucursal, cheque.numero_cheque,
    cheque.numero_cuenta, cheque.cuit_librador, cheque.nombre_librador,
  ].filter((v) => v == null || v === '').length;

  const lowConfidence = [
    cheque.monto_llm_confidence,
    cheque.fecha_emision_llm_confidence,
    cheque.fecha_pago_llm_confidence,
  ].some((c) => c !== null && c < 0.7);

  const borderColor =
    missingCount > 0 ? 'border-red-300 dark:border-red-900'
    : lowConfidence ? 'border-yellow-400 dark:border-yellow-700'
    : 'border-gray-200 dark:border-gray-700';

  return (
    <div
      className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border-2 overflow-hidden
        cursor-pointer transition hover:shadow-md hover:scale-[1.01] ${borderColor}
        flex flex-col h-full`}
      onClick={() => onOpen(cheque)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onOpen(cheque);
        }
      }}
    >
      <div className="bg-gray-100 dark:bg-gray-900 px-3 py-2 flex items-center justify-between text-sm">
        <span className="font-mono text-gray-600 dark:text-gray-400">
          p{cheque.pagina} · ch{cheque.indice_en_pagina}
        </span>
        <div className="flex items-center gap-2 text-xs">
          {!cheque.extracted && (
            <span className="text-blue-600 dark:text-blue-400 animate-pulse">extrayendo...</span>
          )}
          {cheque.extracted && missingCount > 0 && (
            <span className="text-red-600 dark:text-red-400">{missingCount} faltantes</span>
          )}
        </div>
      </div>

      <div className="p-2 bg-gray-50 dark:bg-gray-950">
        <img
          src={imagenUrl(cheque.id)}
          alt={`Cheque p${cheque.pagina}_ch${cheque.indice_en_pagina}`}
          className="w-full h-auto max-h-40 object-contain"
        />
      </div>

      <div className="px-3 py-2 text-sm space-y-1 flex-1">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-gray-500">Monto</span>
          <span className="font-mono font-medium">{formatMonto(cheque.monto)}</span>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-gray-500">F. emisión</span>
          <span className="font-mono">{formatFecha(cheque.fecha_emision)}</span>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-gray-500">F. pago</span>
          <span className="font-mono">{formatFecha(cheque.fecha_pago)}</span>
        </div>
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-xs text-gray-500">Librador</span>
          <span className="truncate text-right" title={cheque.nombre_librador ?? ''}>
            {cheque.nombre_librador ?? '—'}
          </span>
        </div>
      </div>

      <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700 text-xs text-center text-blue-600 dark:text-blue-400">
        Click para revisar y editar →
      </div>
    </div>
  );
}
