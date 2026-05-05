import { useQuery } from '@tanstack/react-query';
import { listRuns } from '../api/client';
import type { Run } from '../api/types';

interface Props {
  activeRunId: number | null;
  onSelect: (run: Run) => void;
}

export function RunsList({ activeRunId, onSelect }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['runs'],
    queryFn: listRuns,
  });

  if (isLoading) return <div className="text-sm text-gray-500">Cargando procesamientos...</div>;
  if (!data || data.length === 0) {
    return <div className="text-sm text-gray-500">Sin procesamientos previos.</div>;
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-200 dark:border-gray-700 text-xs font-semibold uppercase text-gray-500">
        Procesamientos
      </div>
      <ul className="max-h-64 overflow-y-auto">
        {data.map((run) => (
          <li key={run.id}>
            <button
              type="button"
              onClick={() => onSelect(run)}
              className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2
                ${run.id === activeRunId ? 'bg-blue-50 dark:bg-blue-950' : ''}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${
                run.status === 'completed' ? 'bg-green-500'
                : run.status === 'running' ? 'bg-blue-500 animate-pulse'
                : run.status === 'failed' ? 'bg-red-500'
                : 'bg-gray-400'
              }`} />
              <span className="flex-1 truncate" title={run.pdf_filename}>{run.pdf_filename}</span>
              <span className="text-xs text-gray-500">
                {run.total_cheques != null ? `${run.total_cheques} cheques` : '–'}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
