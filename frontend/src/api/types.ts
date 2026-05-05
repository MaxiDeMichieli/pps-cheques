export interface Cheque {
  id: number;
  run_id: number;
  pagina: number;
  indice_en_pagina: number;
  imagen_path: string;

  monto: number | null;
  monto_raw: string;
  monto_score: number;
  monto_llm_confidence: number | null;

  fecha_emision: string | null;
  fecha_emision_raw: string | null;
  fecha_emision_llm_confidence: number | null;

  fecha_pago: string | null;
  fecha_pago_raw: string | null;
  fecha_pago_llm_confidence: number | null;

  sucursal: string | null;
  sucursal_raw: string | null;
  sucursal_score: number;

  numero_sucursal: string | null;
  numero_cheque: string | null;
  numero_cuenta: string | null;

  cuit_librador: string | null;
  nombre_librador: string | null;

  edited_fields: Record<string, string>;
  extracted: boolean;
}

export interface Run {
  id: number;
  created_at: string;
  pdf_filename: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  total_cheques: number | null;
  error: string | null;
}

export interface RunCreated {
  run_id: number;
  pdf_filename: string;
}

export type ChequePatch = Partial<Pick<Cheque,
  | 'monto'
  | 'fecha_emision'
  | 'fecha_pago'
  | 'sucursal'
  | 'numero_sucursal'
  | 'numero_cheque'
  | 'numero_cuenta'
  | 'cuit_librador'
  | 'nombre_librador'
>>;
