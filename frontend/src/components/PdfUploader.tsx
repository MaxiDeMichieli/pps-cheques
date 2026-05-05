import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import { uploadPdf } from '../api/client';
import type { RunCreated } from '../api/types';

interface Props {
  onUploaded: (run: RunCreated) => void;
  disabled?: boolean;
}

export function PdfUploader({ onUploaded, disabled }: Props) {
  const mutation = useMutation({
    mutationFn: uploadPdf,
    onSuccess: onUploaded,
  });

  const onDrop = useCallback(
    (files: File[]) => {
      if (files[0]) mutation.mutate(files[0]);
    },
    [mutation],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: disabled || mutation.isPending,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
        ${isDragActive ? 'border-blue-500 bg-blue-50 dark:bg-blue-950' : 'border-gray-300 dark:border-gray-700'}
        ${(disabled || mutation.isPending) ? 'opacity-50 cursor-not-allowed' : 'hover:border-gray-400'}`}
    >
      <input {...getInputProps()} />
      {mutation.isPending ? (
        <p className="text-gray-500">Subiendo PDF...</p>
      ) : isDragActive ? (
        <p className="text-blue-600">Soltá el PDF acá</p>
      ) : (
        <div>
          <p className="font-medium">Arrastrá un PDF de cheques o clickeá para seleccionar</p>
          <p className="text-sm text-gray-500 mt-1">Solo .pdf</p>
        </div>
      )}
      {mutation.isError && (
        <p className="text-red-600 mt-2 text-sm">
          Error: {(mutation.error as Error).message}
        </p>
      )}
    </div>
  );
}
