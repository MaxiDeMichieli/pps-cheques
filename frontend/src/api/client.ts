import axios from 'axios';
import type { Cheque, ChequePatch, Run, RunCreated } from './types';

const api = axios.create({ baseURL: '/api' });

export async function uploadPdf(file: File): Promise<RunCreated> {
  const form = new FormData();
  form.append('pdf', file);
  const { data } = await api.post<RunCreated>('/runs', form);
  return data;
}

export async function getCheques(runId: number): Promise<Cheque[]> {
  const { data } = await api.get<Cheque[]>(`/runs/${runId}/cheques`);
  return data;
}

export async function patchCheque(id: number, patch: ChequePatch): Promise<Cheque> {
  const { data } = await api.patch<Cheque>(`/cheques/${id}`, patch);
  return data;
}

export async function listRuns(): Promise<Run[]> {
  const { data } = await api.get<Run[]>('/runs');
  return data;
}

export function imagenUrl(chequeId: number): string {
  return `/api/cheques/${chequeId}/imagen`;
}

export function exportUrl(runId: number): string {
  return `/api/runs/${runId}/export.json`;
}
