import type { Cheque } from '../api/types';
import { ChequeCard } from './ChequeCard';

interface Props {
  cheques: Cheque[];
  onOpen: (cheque: Cheque) => void;
}

export function ChequeGrid({ cheques, onOpen }: Props) {
  if (cheques.length === 0) {
    return null;
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {cheques.map((c) => (
        <ChequeCard key={c.id} cheque={c} onOpen={onOpen} />
      ))}
    </div>
  );
}
