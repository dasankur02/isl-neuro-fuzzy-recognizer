import type { Landmark } from '@workspace/api-client-react';

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const fallback: Landmark[] = [
  { x: .47, y: .72, z: 0 }, { x: .31, y: .61, z: 0 }, { x: .22, y: .43, z: 0 }, { x: .19, y: .25, z: 0 }, { x: .2, y: .1, z: 0 },
  { x: .47, y: .52, z: 0 }, { x: .48, y: .32, z: 0 }, { x: .48, y: .14, z: 0 }, { x: .49, y: .05, z: 0 },
  { x: .56, y: .51, z: 0 }, { x: .61, y: .32, z: 0 }, { x: .64, y: .15, z: 0 }, { x: .65, y: .07, z: 0 },
  { x: .64, y: .53, z: 0 }, { x: .73, y: .37, z: 0 }, { x: .78, y: .21, z: 0 }, { x: .82, y: .14, z: 0 },
  { x: .7, y: .6, z: 0 }, { x: .83, y: .5, z: 0 }, { x: .91, y: .42, z: 0 }, { x: .94, y: .37, z: 0 },
];

export function LandmarkCanvas({ landmarks, active = false }: { landmarks?: Landmark[]; active?: boolean }) {
  const points = landmarks?.length === 21 ? landmarks : fallback;
  const scaled = points.map((point) => ({ x: 28 + point.x * 244, y: 20 + point.y * 190 }));
  return (
    <div className="signal-grid relative aspect-[1.24] w-full overflow-hidden border border-[#5b6380]/35 bg-[#252c4a]" data-testid="visual-landmark-canvas">
      <div className="absolute left-4 top-3 font-mono text-[9px] uppercase tracking-[0.16em] text-[#f5cf70]/60">LANDMARK FIELD / 21 NODES</div>
      <div className="absolute right-4 top-3 flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.16em] text-white/40"><span className={`size-1.5 rounded-full ${active ? 'animate-pulse bg-[#f6c341]' : 'bg-[#91a0bf]'}`} />{active ? 'streaming' : 'captured'}</div>
      <svg viewBox="0 0 300 230" className="absolute inset-0 h-full w-full" aria-label="Hand landmark visualization">
        <g opacity=".55" stroke="#9ee0d8" strokeWidth="1.4" fill="none">
          {connections.map(([from, to]) => <line key={`${from}-${to}`} x1={scaled[from].x} y1={scaled[from].y} x2={scaled[to].x} y2={scaled[to].y} />)}
        </g>
        <g>
          {scaled.map((point, index) => <circle key={index} cx={point.x} cy={point.y} r={index === 0 ? 5 : 3.4} fill={index === 0 ? '#f6c341' : '#b8eee2'} stroke="#25304d" strokeWidth="1.5" />)}
        </g>
      </svg>
      <div className="scan-line pointer-events-none absolute inset-x-0 top-0 h-8 bg-gradient-to-b from-transparent via-[#f6c341]/20 to-transparent" />
      <div className="absolute bottom-3 left-4 font-mono text-[9px] text-white/35">x: normalized · y: normalized · z: depth</div>
    </div>
  );
}