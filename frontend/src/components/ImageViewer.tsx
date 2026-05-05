import { useEffect, useLayoutEffect, useRef, useState } from 'react';

interface View {
  zoom: number;
  panX: number;
  panY: number;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 6;
const ZOOM_STEP = 1.25;

interface Props {
  src: string;
  alt: string;
  /**
   * Cambiar este valor fuerza un reset (ej. al cambiar el cheque mostrado).
   */
  resetKey?: string | number;
}

/**
 * Visor de imagen con zoom hacia el cursor (rueda del mouse) y pan por
 * arrastre. Internamente usa transform: translate(...) scale(...) sobre la
 * imagen, con el contenedor en overflow:hidden.
 */
export function ImageViewer({ src, alt, resetKey }: Props) {
  const [view, setView] = useState<View>({ zoom: 1, panX: 0, panY: 0 });
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef(view);
  viewRef.current = view;

  // Reset al cambiar el cheque (o cualquier resetKey)
  useLayoutEffect(() => {
    setView({ zoom: 1, panX: 0, panY: 0 });
  }, [resetKey]);

  // Wheel zoom — listener no-passive para poder cancelar el scroll de página
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const mx = e.clientX - cx;
      const my = e.clientY - cy;

      const factor = Math.exp(-e.deltaY * 0.0015);
      const v = viewRef.current;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, v.zoom * factor));
      const realFactor = newZoom / v.zoom;
      setView({
        zoom: newZoom,
        panX: v.panX * realFactor + mx * (1 - realFactor),
        panY: v.panY * realFactor + my * (1 - realFactor),
      });
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // Drag pan — pegamos los listeners de move/up al window mientras drag está activo
  useEffect(() => {
    if (!dragging) return;
    let lastX = 0;
    let lastY = 0;
    let inited = false;

    const onMove = (e: MouseEvent) => {
      if (!inited) {
        lastX = e.clientX;
        lastY = e.clientY;
        inited = true;
        return;
      }
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      const v = viewRef.current;
      setView({ zoom: v.zoom, panX: v.panX + dx, panY: v.panY + dy });
    };
    const onUp = () => setDragging(false);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging]);

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    setDragging(true);
  };

  const zoomBy = (mult: number) => {
    setView((v) => {
      const nz = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, v.zoom * mult));
      const realFactor = nz / v.zoom;
      return { zoom: nz, panX: v.panX * realFactor, panY: v.panY * realFactor };
    });
  };

  const reset = () => setView({ zoom: 1, panX: 0, panY: 0 });

  return (
    <div
      ref={containerRef}
      className={`bg-gray-100 dark:bg-gray-950 relative overflow-hidden p-4 flex items-center justify-center
        ${dragging ? 'cursor-grabbing' : 'cursor-grab'}`}
      onMouseDown={onMouseDown}
    >
      <img
        src={src}
        alt={alt}
        draggable={false}
        style={{
          transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`,
          transformOrigin: 'center center',
        }}
        className="max-w-full max-h-full h-auto select-none pointer-events-none"
      />

      <div className="absolute bottom-3 right-3 flex gap-1 bg-white/90 dark:bg-gray-800/90 rounded-md shadow p-1 text-sm">
        <button
          type="button"
          onClick={() => zoomBy(1 / ZOOM_STEP)}
          className="px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          title="Zoom out"
        >
          −
        </button>
        <span className="px-2 py-1 tabular-nums select-none">
          {Math.round(view.zoom * 100)}%
        </span>
        <button
          type="button"
          onClick={() => zoomBy(ZOOM_STEP)}
          className="px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          title="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={reset}
          className="px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          title="Reset"
        >
          ⟲
        </button>
      </div>

      <div className="absolute top-3 left-3 bg-white/80 dark:bg-gray-800/80 rounded px-2 py-1 text-xs text-gray-600 dark:text-gray-400 pointer-events-none select-none">
        Rueda: zoom · Arrastrar: mover
      </div>
    </div>
  );
}
