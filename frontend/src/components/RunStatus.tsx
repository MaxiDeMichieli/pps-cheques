import type { RunStreamState } from '../hooks/useRunStream';

interface Props {
  state: RunStreamState;
  pdfFilename: string | null;
}

export function RunStatus({ state, pdfFilename }: Props) {
  const total = state.cheques.size;
  const extraidos = Array.from(state.cheques.values()).filter((c) => c.extracted).length;

  const label =
    state.status === 'connecting' ? 'Conectando...'
    : state.status === 'streaming' ? `Procesando ${pdfFilename ?? ''}`
    : state.status === 'completed' ? (state.durationS != null ? `Listo (${state.durationS}s)` : 'Listo')
    : state.status === 'error' ? `Error: ${state.error}`
    : '';

  if (state.status === 'idle') return null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-700 flex items-center gap-3 text-sm">
      <span className={`inline-block w-2 h-2 rounded-full ${
        state.status === 'streaming' ? 'bg-blue-500 animate-pulse'
        : state.status === 'completed' ? 'bg-green-500'
        : state.status === 'error' ? 'bg-red-500'
        : 'bg-gray-400'
      }`} />
      <span className="font-medium">{label}</span>
      {state.totalPages !== null && state.totalPages > 0 && (
        <span className="text-gray-500">· {state.totalPages} páginas</span>
      )}
      <span className="text-gray-500">· {extraidos}/{total} cheques extraídos</span>
    </div>
  );
}
