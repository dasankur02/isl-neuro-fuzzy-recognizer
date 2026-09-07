import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Check, CircleDot, Clock3, Crosshair, Layers3, RefreshCw, ScanFace, Sparkles, Zap } from 'lucide-react';
import { getGetRecognitionDemoQueryKey, useClassifyRecognition, useGetRecognitionDemo } from '@workspace/api-client-react';
import type { RecognitionResult } from '@workspace/api-client-react';
import { DataState, PageHeader } from '@/components/app-shell';
import { LandmarkCanvas } from '@/components/landmark-canvas';

const emptyProbabilities = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'].map((label) => ({ label, probability: 0 }));
type CameraLandmark = { x: number; y: number; z: number };
type CameraStatus = 'starting' | 'loading-model' | 'ready' | 'no-hand' | 'error';

type HandLandmarkerLike = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => { landmarks?: CameraLandmark[][] };
  close: () => void;
};

type VisionModule = {
  FilesetResolver: { forVisionTasks: (wasmPath: string) => Promise<unknown> };
  HandLandmarker: { createFromOptions: (resolver: unknown, options: Record<string, unknown>) => Promise<HandLandmarkerLike> };
};

const VISION_BUNDLE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
const VISION_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

function MetricTile({ label, value, suffix, detail }: { label: string; value: string; suffix?: string; detail: string }) {
  return <div className="border-l-2 border-primary bg-card px-4 py-3" data-testid={`metric-${label.toLowerCase().replaceAll(' ', '-')}`}>
    <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-muted-foreground">{label}</div>
    <div className="mt-1 font-display text-2xl font-bold tracking-tight text-foreground">{value}<span className="ml-1 font-mono text-[11px] font-medium text-muted-foreground">{suffix}</span></div>
    <div className="mt-1 text-[10px] text-muted-foreground">{detail}</div>
  </div>;
}

function PipelineStep({ number, title, description, active, complete }: { number: string; title: string; description: string; active?: boolean; complete?: boolean }) {
  return <div className={`relative flex gap-3 border-l px-4 py-3 transition-colors ${active ? 'border-primary bg-primary/10' : 'border-border bg-card/50'}`} data-testid={`pipeline-step-${number}`}>
    <span className={`mt-0.5 flex size-6 shrink-0 items-center justify-center font-mono text-[10px] ${active ? 'bg-primary text-primary-foreground' : complete ? 'bg-accent text-accent-foreground' : 'border border-border text-muted-foreground'}`}>{complete ? <Check size={13} /> : number}</span>
    <div><div className="text-[12px] font-bold text-foreground">{title}</div><div className="mt-0.5 text-[10px] leading-4 text-muted-foreground">{description}</div></div>
    {active && <span className="absolute right-3 top-3 size-1.5 animate-pulse bg-primary" />}
  </div>;
}

function CameraPreview({ onDetect }: { onDetect: (landmarks: CameraLandmark[]) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const onDetectRef = useRef(onDetect);
  const [status, setStatus] = useState<CameraStatus>('starting');
  const [lastDetected, setLastDetected] = useState<CameraLandmark[] | null>(null);

  useEffect(() => {
    onDetectRef.current = onDetect;
  }, [onDetect]);

  useEffect(() => {
    let stream: MediaStream | undefined;
    let detector: HandLandmarkerLike | undefined;
    let animationFrame = 0;
    let cancelled = false;

    const start = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('error');
        return;
      }
      try {
        setStatus('loading-model');
        const vision = await import(/* @vite-ignore */ VISION_BUNDLE_URL) as unknown as VisionModule;
        const resolver = await vision.FilesetResolver.forVisionTasks(VISION_WASM_URL);
        detector = await vision.HandLandmarker.createFromOptions(resolver, {
          baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
        });
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
          audio: false,
        });
        if (cancelled || !videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setStatus('ready');

        const detectFrame = () => {
          if (cancelled || !detector || !videoRef.current) return;
          const video = videoRef.current;
          if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime > 0) {
            const result = detector.detectForVideo(video, performance.now());
            const landmarks = result.landmarks?.[0] ?? null;
            setLastDetected(landmarks);
            if (landmarks) {
              onDetectRef.current(landmarks);
              setStatus('ready');
            } else {
              setStatus('no-hand');
            }
          }
          animationFrame = requestAnimationFrame(detectFrame);
        };
        animationFrame = requestAnimationFrame(detectFrame);
      } catch {
        setStatus('error');
      }
    };
    void start();
    return () => {
      cancelled = true;
      cancelAnimationFrame(animationFrame);
      detector?.close();
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  if (status === 'error') {
    return <div className="flex aspect-[1.24] flex-col items-center justify-center gap-2 border border-[#ee765e]/50 bg-[#ee765e]/10 p-6 text-center text-sm text-[#8e2c1e]">
      <ScanFace size={24} />
      <b>Hand tracking could not start</b>
      <span className="max-w-xs text-xs leading-5">Allow camera access and reload the preview so the browser can load the hand-landmark model.</span>
    </div>;
  }

  return <div className="relative aspect-[1.24] overflow-hidden bg-sidebar/90" data-testid="camera-preview">
    <video ref={videoRef} muted playsInline className="size-full -scale-x-100 object-cover opacity-80" />
    {lastDetected && <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 size-full -scale-x-100">
      {lastDetected.map((point, index) => <circle key={index} cx={point.x} cy={point.y} r={index === 0 ? 0.018 : 0.012} fill={index === 0 ? '#f6c341' : '#b8eee2'} stroke="#25304d" strokeWidth=".004" />)}
    </svg>}
    <div className="absolute inset-0 border border-primary/50" />
    <div className="absolute left-4 top-3 flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.15em] text-primary">
      <span className={`size-1.5 ${status === 'no-hand' ? 'bg-[#ee765e]' : 'animate-pulse bg-primary'}`} />{status === 'loading-model' ? 'loading hand tracker' : status === 'no-hand' ? 'show one hand to detect' : 'camera / 640×480 / 30 fps target'}
    </div>
    <div className="absolute inset-x-5 bottom-4 border border-primary/35 bg-sidebar/80 px-3 py-2 text-[10px] leading-4 text-sidebar-foreground/75">
      {lastDetected ? '21 landmarks detected · live recognition is updating' : 'Point one hand at the camera. The browser is loading the 21-point hand tracker.'}
    </div>
  </div>;
}

export default function Workbench() {
  const demoQuery = useGetRecognitionDemo({ query: { queryKey: getGetRecognitionDemoQueryKey() } });
  const classify = useClassifyRecognition();
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [feed, setFeed] = useState('demo');
  const [capturedAt, setCapturedAt] = useState('waiting for input');
  const cameraSequence = useRef<CameraLandmark[][]>([]);
  const lastCameraRun = useRef(0);
  const classifyPending = useRef(false);

  useEffect(() => {
    if (demoQuery.data) setCapturedAt('just now · synthetic sequence');
  }, [demoQuery.data]);

  const demo = demoQuery.data;
  const probabilities = result?.probabilities ?? emptyProbabilities;
  const sortedProbabilities = useMemo(() => [...probabilities].sort((a, b) => b.probability - a.probability), [probabilities]);

  function runRecognition() {
    const sequence = feed === 'camera' ? cameraSequence.current : demo?.sequence;
    const landmarks = feed === 'camera' ? sequence?.at(-1) : demo?.landmarks;
    if (!landmarks) return;
    setCapturedAt('processing sequence…');
    classifyPending.current = true;
    classify.mutate({ data: { landmarks, sequence } }, {
      onSuccess: (response) => { setResult(response); setCapturedAt(feed === 'camera' ? 'just now · camera classified' : 'just now · classified'); classifyPending.current = false; },
      onError: () => { classifyPending.current = false; },
    });
  }

  function handleCameraDetect(landmarks: CameraLandmark[]) {
    cameraSequence.current = [...cameraSequence.current.slice(-11), landmarks];
    const now = Date.now();
    if (now - lastCameraRun.current < 650 || classifyPending.current) return;
    lastCameraRun.current = now;
    runRecognition();
  }

  return <div className="min-h-[100dvh]">
    <PageHeader eyebrow="Live recognition / spatial + temporal" title="Signal workbench" description="Trace a hand through the interpretable pipeline. The camera path tracks 21 landmarks in the browser and sends a short temporal window to the hybrid model." action={
      <div className="flex items-center gap-2 border border-accent/70 bg-accent/35 px-3 py-2 text-xs font-semibold text-accent-foreground"><CircleDot size={14} className="text-[#178c83]" />{feed === 'camera' ? 'Camera tracker ready' : 'Synthetic stream ready'}</div>
    } />
    <div className="mx-auto max-w-[1500px] space-y-5 px-5 py-6 sm:px-8 lg:space-y-6 lg:px-10 lg:py-8">
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(330px,.7fr)]">
        <section className="panel-shadow overflow-hidden border border-border bg-card stagger-in" data-testid="section-input-signal">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
            <div><div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground"><ScanFace size={14} className="text-primary" />Input signal</div><div className="mt-1 text-xs text-muted-foreground">Landmark geometry / frame 01 of {demo?.sequence?.length ?? 0}</div></div>
            <div className="flex items-center gap-1 border border-border bg-background p-1 font-mono text-[10px] uppercase"><button onClick={() => setFeed('demo')} className={`px-3 py-1.5 transition ${feed === 'demo' ? 'bg-sidebar text-sidebar-foreground' : 'text-muted-foreground hover:text-foreground'}`} data-testid="button-input-demo">Demo sequence</button><button onClick={() => setFeed('camera')} className={`px-3 py-1.5 transition ${feed === 'camera' ? 'bg-sidebar text-sidebar-foreground' : 'text-muted-foreground hover:text-foreground'}`} data-testid="button-input-camera">Camera</button></div>
          </div>
          <div className="grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_180px]">
            {feed === 'camera' ? <CameraPreview onDetect={handleCameraDetect} /> : demoQuery.isLoading ? <div className="aspect-[1.24] animate-pulse bg-sidebar/90" data-testid="skeleton-landmark-canvas" /> : demoQuery.isError ? <DataState type="error" message="The demo signal could not be loaded." action={<button onClick={() => demoQuery.refetch()} className="border border-border px-3 py-2 text-xs font-bold hover:bg-muted" data-testid="button-retry-demo">Try again</button>} /> : <LandmarkCanvas landmarks={demo?.landmarks} active={classify.isPending} />}
            <div className="flex flex-col justify-between gap-5">
              <div className="space-y-3 font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground"><div className="flex justify-between border-b border-border pb-2"><span>source</span><b className="font-medium text-foreground">{feed === 'demo' ? 'synthetic' : 'live camera'}</b></div><div className="flex justify-between border-b border-border pb-2"><span>points</span><b className="font-medium text-foreground">{feed === 'camera' ? 'live / 21' : `${demo?.landmarks?.length ?? '—'} / 21`}</b></div><div className="flex justify-between border-b border-border pb-2"><span>window</span><b className="font-medium text-foreground">{feed === 'camera' ? `${cameraSequence.current.length || '—'} frames` : `${demo?.sequence?.length ?? '—'} frames`}</b></div><div className="flex justify-between"><span>captured</span><b className="text-right font-medium text-foreground">{capturedAt}</b></div></div>
              <div><button onClick={runRecognition} disabled={(!demo && feed === 'demo') || (feed === 'camera' && !cameraSequence.current.length) || classify.isPending} className="group flex w-full items-center justify-between bg-primary px-4 py-3 text-left text-xs font-bold text-primary-foreground transition hover:bg-[#f8d15f] disabled:cursor-not-allowed disabled:opacity-50" data-testid="button-run-recognition">{classify.isPending ? <><span className="animate-pulse">Running stages…</span><Zap size={15} /></> : <><span>Run recognition</span><ArrowRight size={15} className="transition-transform group-hover:translate-x-1" /></>}</button><button onClick={() => { setResult(null); cameraSequence.current = []; demoQuery.refetch(); }} className="mt-2 flex w-full items-center justify-center gap-2 py-2 text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground transition hover:text-foreground" data-testid="button-refresh-signal"><RefreshCw size={12} /> Refresh signal</button></div>
            </div>
          </div>
        </section>

        <section className="panel-shadow border border-border bg-sidebar text-sidebar-foreground stagger-in stagger-1" data-testid="section-recognition-result">
          <div className="flex items-center justify-between border-b border-sidebar-border px-5 py-4"><div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-sidebar-foreground/65"><Crosshair size={14} className="text-primary" />Recognition output</div><span className="font-mono text-[10px] text-sidebar-foreground/40">{result ? 'RESULT / 001' : 'AWAITING RUN'}</span></div>
          <div className="p-5">
            {classify.isError && <div className="mb-4 border border-[#ee765e]/50 bg-[#ee765e]/10 px-3 py-2 text-xs text-[#ffc0b2]" data-testid="status-recognition-error">Classification failed. Check the signal and run again.</div>}
            <div className="flex items-end justify-between border-b border-sidebar-border pb-5"><div><div className="font-mono text-[10px] uppercase tracking-[0.18em] text-sidebar-foreground/45">Predicted digit</div><div className="mt-1 font-display text-[90px] font-bold leading-none tracking-[-0.1em] text-primary" data-testid="text-predicted-digit">{result?.label ?? '—'}</div></div><div className="text-right"><div className="font-mono text-[10px] uppercase tracking-[0.12em] text-sidebar-foreground/45">Confidence</div><div className="mt-1 font-display text-3xl font-bold" data-testid="text-recognition-confidence">{result ? `${(result.confidence * 100).toFixed(1)}%` : '—'}</div></div></div>
            <div className="mt-5 space-y-2"><div className="flex justify-between font-mono text-[9px] uppercase tracking-[0.12em] text-sidebar-foreground/45"><span>Posterior distribution</span><span>p(label | x)</span></div>{sortedProbabilities.slice(0, 5).map((item) => <div key={item.label} className="flex items-center gap-3" data-testid={`probability-${item.label}`}><span className={`w-4 font-mono text-[11px] ${item.label === result?.label ? 'font-bold text-primary' : 'text-sidebar-foreground/60'}`}>{item.label}</span><div className="h-1.5 flex-1 bg-sidebar-accent"><div className={`h-full transition-all duration-700 ${item.label === result?.label ? 'bg-primary' : 'bg-[#91a0bf]/60'}`} style={{ width: `${Math.max(item.probability * 100, item.probability ? 2 : 0)}%` }} /></div><span className="w-10 text-right font-mono text-[10px] text-sidebar-foreground/55">{(item.probability * 100).toFixed(1)}%</span></div>)}</div>
          </div>
          <div className="grid grid-cols-3 border-t border-sidebar-border"><div className="p-4"><div className="font-mono text-[9px] text-sidebar-foreground/40">STAGE</div><div className="mt-1 text-xs font-bold" data-testid="text-recognition-stage">{result?.stage ?? '—'}</div></div><div className="border-x border-sidebar-border p-4"><div className="font-mono text-[9px] text-sidebar-foreground/40">LATENCY</div><div className="mt-1 text-xs font-bold" data-testid="text-recognition-latency">{result ? `${result.latencyMs} ms` : '—'}</div></div><div className="p-4"><div className="font-mono text-[9px] text-sidebar-foreground/40">STATUS</div><div className="mt-1 flex items-center gap-1 text-xs font-bold text-[#b8eee2]"><span className={`size-1.5 ${result ? 'bg-[#b8eee2]' : 'bg-sidebar-foreground/30'}`} />{result ? 'stable' : 'idle'}</div></div></div>
        </section>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(340px,.8fr)]">
        <section className="border border-border bg-card p-5 sm:p-6 stagger-in stagger-2" data-testid="section-model-pipeline">
          <div className="mb-5 flex items-start justify-between"><div><div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground"><Layers3 size={14} className="text-primary" />Current model pipeline</div><p className="mt-1 text-xs text-muted-foreground">A readable path from pose to prediction.</p></div><span className="font-mono text-[10px] text-muted-foreground">HYBRID / 02 STAGES</span></div>
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4"><PipelineStep number="01" title="Landmark capture" description="21 keypoints · normalized xyz" complete /><PipelineStep number="02" title="Feature extraction" description="count · spread · orientation" complete /><PipelineStep number="03" title="ANFIS inference" description="3-feature fuzzy rule base" active={classify.isPending} complete={!!result} /><PipelineStep number="04" title="GRU temporal read" description="sequence context · 60 frames" active={!!result && !classify.isPending} /></div>
        </section>
        <section className="grid grid-cols-3 gap-3 stagger-in stagger-3" data-testid="section-feature-metrics">
          <MetricTile label="Finger count" value={result ? `${result.features.fingerCount}` : '—'} detail="detected ext." />
          <MetricTile label="Spread" value={result ? result.features.spread.toFixed(2) : '—'} detail="normalized" />
          <MetricTile label="Orientation" value={result ? result.features.orientation.toFixed(2) : '—'} suffix="norm." detail="wrist axis" />
        </section>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4 font-mono text-[10px] uppercase tracking-[0.13em] text-muted-foreground"><span className="flex items-center gap-2"><Clock3 size={13} /> Live session / local view</span><span className="flex items-center gap-2"><Sparkles size={13} className="text-primary" /> Explainability features are model inputs, not post-hoc guesses</span></div>
    </div>
  </div>;
}