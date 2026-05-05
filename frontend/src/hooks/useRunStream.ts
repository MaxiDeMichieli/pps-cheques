import { useCallback, useEffect, useState } from 'react';
import type { Cheque } from '../api/types';
import { getCheques } from '../api/client';

export interface RunStreamState {
  status: 'idle' | 'connecting' | 'streaming' | 'completed' | 'error';
  totalPages: number | null;
  cheques: Map<number, Cheque>;
  error: string | null;
  durationS: number | null;
}

export interface RunStream {
  state: RunStreamState;
  applyLocalPatch: (id: number, patch: Partial<Cheque>) => void;
}

const initialState: RunStreamState = {
  status: 'idle',
  totalPages: null,
  cheques: new Map(),
  error: null,
  durationS: null,
};

interface ChequeDetectedPayload {
  cheque_id: number;
  pagina: number;
  indice: number;
  imagen_url: string;
}

/**
 * Conecta a /api/runs/{id}/events vía EventSource y mantiene en estado el
 * progreso del run (cheques detectados/extraídos, total páginas, status).
 */
/**
 * Si ``live`` es true, abre SSE para escuchar el progreso del run.
 * Si es false (run ya completado), solo carga los cheques vía REST.
 */
export function useRunStream(runId: number | null, live: boolean = true): RunStream {
  const [state, setState] = useState<RunStreamState>(initialState);

  const applyLocalPatch = useCallback((id: number, patch: Partial<Cheque>) => {
    setState((s) => {
      const existing = s.cheques.get(id);
      if (!existing) return s;
      const cheques = new Map(s.cheques);
      cheques.set(id, { ...existing, ...patch });
      return { ...s, cheques };
    });
  }, []);

  useEffect(() => {
    if (runId == null) {
      setState(initialState);
      return;
    }

    if (!live) {
      setState({ ...initialState, status: 'connecting' });
      getCheques(runId)
        .then((cheques) => {
          const map = new Map<number, Cheque>();
          for (const c of cheques) map.set(c.id, c);
          setState({
            status: 'completed',
            totalPages: null,
            cheques: map,
            error: null,
            durationS: null,
          });
        })
        .catch((err: Error) =>
          setState((s) => ({ ...s, status: 'error', error: err.message })),
        );
      return;
    }

    setState({ ...initialState, status: 'connecting' });
    let cancelled = false;

    // Sembramos el estado con los cheques ya persistidos en DB. Esto cubre el
    // caso de reconexión: si ya hubo una sesión SSE previa que consumió
    // eventos y se desconectó, esos eventos no se vuelven a emitir, pero los
    // cheques quedaron persistidos y los recuperamos vía REST.
    getCheques(runId)
      .then((cheques) => {
        if (cancelled) return;
        setState((s) => {
          const map = new Map(s.cheques);
          for (const c of cheques) {
            if (!map.has(c.id)) map.set(c.id, c);
          }
          return { ...s, cheques: map };
        });
      })
      .catch(() => { /* silencioso: SSE seguirá poblando */ });

    const es = new EventSource(`/api/runs/${runId}/events`);

    es.addEventListener('open', () => {
      setState((s) => ({ ...s, status: 'streaming' }));
    });

    es.addEventListener('pdf_loaded', (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as { total_pages: number };
      setState((s) => ({ ...s, totalPages: data.total_pages }));
    });

    es.addEventListener('cheque_detected', (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as ChequeDetectedPayload;
      setState((s) => {
        const cheques = new Map(s.cheques);
        if (!cheques.has(data.cheque_id)) {
          cheques.set(data.cheque_id, makeStubCheque(runId, data));
        }
        return { ...s, cheques };
      });
    });

    es.addEventListener('cheque_extracted', (ev) => {
      const cheque = JSON.parse((ev as MessageEvent).data) as Cheque;
      setState((s) => {
        const cheques = new Map(s.cheques);
        cheques.set(cheque.id, cheque);
        return { ...s, cheques };
      });
    });

    es.addEventListener('run_completed', (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        total_cheques: number;
        duration_s: number;
        status: string;
      };
      setState((s) => ({
        ...s,
        status: data.status === 'failed' ? 'error' : 'completed',
        durationS: data.duration_s,
      }));
      es.close();
    });

    es.addEventListener('error', (ev) => {
      const msg = ev instanceof MessageEvent && ev.data
        ? (JSON.parse(ev.data) as { message: string }).message
        : 'Error de conexión SSE';
      setState((s) => ({ ...s, status: 'error', error: msg }));
    });

    return () => {
      cancelled = true;
      es.close();
    };
  }, [runId, live]);

  return { state, applyLocalPatch };
}

function makeStubCheque(runId: number, p: ChequeDetectedPayload): Cheque {
  return {
    id: p.cheque_id,
    run_id: runId,
    pagina: p.pagina,
    indice_en_pagina: p.indice,
    imagen_path: '',
    monto: null,
    monto_raw: '',
    monto_score: 0,
    monto_llm_confidence: null,
    fecha_emision: null,
    fecha_emision_raw: null,
    fecha_emision_llm_confidence: null,
    fecha_pago: null,
    fecha_pago_raw: null,
    fecha_pago_llm_confidence: null,
    sucursal: null,
    sucursal_raw: null,
    sucursal_score: 0,
    numero_sucursal: null,
    numero_cheque: null,
    numero_cuenta: null,
    cuit_librador: null,
    nombre_librador: null,
    edited_fields: {},
    extracted: false,
  };
}
