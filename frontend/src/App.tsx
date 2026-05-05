import { useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { PdfUploader } from './components/PdfUploader';
import { ChequeGrid } from './components/ChequeGrid';
import { ChequeDetail } from './components/ChequeDetail';
import { RunStatus } from './components/RunStatus';
import { RunsList } from './components/RunsList';
import { useRunStream } from './hooks/useRunStream';
import { exportUrl } from './api/client';
import type { Cheque, Run, RunCreated } from './api/types';
import './App.css';

interface ActiveRun {
  id: number;
  pdf_filename: string;
  live: boolean;
}

function App() {
  const [activeRun, setActiveRun] = useState<ActiveRun | null>(null);
  const [selectedChequeId, setSelectedChequeId] = useState<number | null>(null);
  const queryClient = useQueryClient();
  const { state, applyLocalPatch } = useRunStream(
    activeRun?.id ?? null,
    activeRun?.live ?? true,
  );

  // Cuando termina (o falla) el procesamiento, refrescar el listado lateral
  // para que el status pase de running → completed/failed con total_cheques.
  useEffect(() => {
    if (state.status === 'completed' || state.status === 'error') {
      queryClient.invalidateQueries({ queryKey: ['runs'] });
    }
  }, [state.status, queryClient]);

  const cheques: Cheque[] = useMemo(
    () =>
      Array.from(state.cheques.values()).sort(
        (a, b) =>
          a.pagina - b.pagina ||
          a.indice_en_pagina - b.indice_en_pagina,
      ),
    [state.cheques],
  );

  const selectedCheque = selectedChequeId != null
    ? state.cheques.get(selectedChequeId) ?? null
    : null;

  const onUploaded = (run: RunCreated) => {
    setActiveRun({ id: run.run_id, pdf_filename: run.pdf_filename, live: true });
    queryClient.invalidateQueries({ queryKey: ['runs'] });
  };

  const onSelectRun = (run: Run) => {
    setActiveRun({
      id: run.id,
      pdf_filename: run.pdf_filename,
      live: run.status === 'pending' || run.status === 'running',
    });
  };

  const onOpenCheque = (cheque: Cheque) => {
    if (!cheque.extracted) return;
    setSelectedChequeId(cheque.id);
  };

  const onCloseModal = () => setSelectedChequeId(null);

  const onChequeSaved = (updated: Cheque) => {
    applyLocalPatch(updated.id, updated);
  };

  const isProcessing = state.status === 'streaming' || state.status === 'connecting';

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto">
      <header className="mb-6 flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold mb-1">Procesador de Cheques</h1>
          <p className="text-sm text-gray-500">
            Cargá un PDF y revisá los cheques en vivo.
          </p>
        </div>
        {activeRun && (
          <button
            type="button"
            onClick={() => setActiveRun(null)}
            className="text-sm text-gray-500 hover:text-gray-900 dark:hover:text-gray-100"
            disabled={isProcessing}
          >
            Cerrar procesamiento
          </button>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-6">
        <main className="space-y-4">
          <PdfUploader onUploaded={onUploaded} disabled={isProcessing} />

          {activeRun && (
            <div className="flex items-center gap-3">
              <RunStatus state={state} pdfFilename={activeRun.pdf_filename} />
              {state.status === 'completed' && (
                <a
                  href={exportUrl(activeRun.id)}
                  download
                  className="text-sm bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded"
                >
                  Exportar JSON
                </a>
              )}
            </div>
          )}

          <ChequeGrid cheques={cheques} onOpen={onOpenCheque} />
        </main>

        <aside className="space-y-4">
          <RunsList activeRunId={activeRun?.id ?? null} onSelect={onSelectRun} />
        </aside>
      </div>

      {selectedCheque && (
        <ChequeDetail
          cheque={selectedCheque}
          onClose={onCloseModal}
          onSaved={onChequeSaved}
        />
      )}
    </div>
  );
}

export default App;
