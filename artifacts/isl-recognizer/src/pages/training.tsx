import { useMemo, useState } from 'react';
import { Check, Cpu, Database, FileUp, Gauge, LineChart, Play, RotateCcw, SlidersHorizontal, Sparkles, UploadCloud, X } from 'lucide-react';
import { useRunTraining } from '@workspace/api-client-react';
import type { TrainingResult, TrainingStep } from '@workspace/api-client-react';
import { PageHeader } from '@/components/app-shell';
import {
  DIGITS,
  extractMediaSample,
  parseStructuredText,
  type Digit,
  type TrainingSample,
} from '@/lib/training-parser';

type RunRecord = TrainingResult & { ranAt: string };

function LossChart({ steps }: { steps: TrainingStep[] }) {
  const chartSteps = steps.length ? steps : Array.from({ length: 8 }, (_, index) => ({ epoch: index + 1, stage: index < 4 ? 'ANFIS' : 'GRU', loss: .42 - index * .025, accuracy: .6 + index * .035 }));
  const maxLoss = Math.max(...chartSteps.map((step) => step.loss), .5);
  const minLoss = Math.min(...chartSteps.map((step) => step.loss), 0);
  const points = chartSteps.map((step, index) => `${(index / Math.max(chartSteps.length - 1, 1)) * 100},${96 - ((step.loss - minLoss) / Math.max(maxLoss - minLoss, .01)) * 76}`).join(' ');
  return <div className="relative h-[190px] border border-border bg-background p-4" data-testid="chart-training-loss">
    <div className="absolute inset-x-4 top-4 flex justify-between font-mono text-[9px] uppercase tracking-[0.13em] text-muted-foreground"><span>Loss trajectory</span><span>{steps.length ? 'measured run' : 'awaiting dataset'}</span></div>
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-x-4 bottom-7 top-9 h-[130px] w-[calc(100%-2rem)] overflow-visible">
      <line x1="0" y1="20" x2="100" y2="20" stroke="hsl(var(--border))" strokeWidth=".5" /><line x1="0" y1="58" x2="100" y2="58" stroke="hsl(var(--border))" strokeWidth=".5" /><line x1="0" y1="96" x2="100" y2="96" stroke="hsl(var(--border))" strokeWidth=".5" />
      <polyline fill="none" stroke="#178c83" strokeWidth="1.2" points={points} vectorEffect="non-scaling-stroke" />
      {chartSteps.map((step, index) => <circle key={index} cx={(index / Math.max(chartSteps.length - 1, 1)) * 100} cy={96 - ((step.loss - minLoss) / Math.max(maxLoss - minLoss, .01)) * 76} r="1.8" fill="#f6c341" vectorEffect="non-scaling-stroke" />)}
    </svg>
    <div className="absolute bottom-3 inset-x-4 flex justify-between font-mono text-[9px] text-muted-foreground"><span>epoch 01</span><span>epoch {String(chartSteps.at(-1)?.epoch ?? 8).padStart(2, '0')}</span></div>
  </div>;
}

function RunSummary({ run }: { run: RunRecord }) {
  const confusion = run.confusionCases ?? [];
  return <div className="border border-accent/80 bg-accent/25 p-4" data-testid="status-training-success">
    <div className="flex items-center gap-2 text-xs font-bold text-accent-foreground"><span className="flex size-5 items-center justify-center bg-accent text-accent-foreground"><Check size={13} /></span>Training run complete · model {run.modelVersion}</div>
    <div className="mt-3 grid grid-cols-2 gap-y-3 text-[11px] sm:grid-cols-5">
      <div><div className="font-mono text-[9px] uppercase text-muted-foreground">validation</div><b>{((run.validationAccuracy ?? 0) * 100).toFixed(1)}%</b></div>
      <div><div className="font-mono text-[9px] uppercase text-muted-foreground">held out</div><b>{run.validationSamples ?? 0}</b></div>
      <div><div className="font-mono text-[9px] uppercase text-muted-foreground">train</div><b>{run.trainSamples ?? 0}</b></div>
      <div><div className="font-mono text-[9px] uppercase text-muted-foreground">rules</div><b>{run.anfisRules}</b></div>
      <div><div className="font-mono text-[9px] uppercase text-muted-foreground">confusions</div><b>{confusion.reduce((sum, item) => sum + item.count, 0)}</b></div>
    </div>
    {confusion.length > 0 && <div className="mt-4 border-t border-accent/40 pt-3 text-[10px] text-muted-foreground"><span className="font-mono uppercase tracking-[0.12em]">Confusion cases · </span>{confusion.slice(0, 3).map((item) => `${item.actual} → ${item.predicted} (${item.count})`).join(' · ')}</div>}
  </div>;
}

export default function Training() {
  const training = useRunTraining();
  const [epochs, setEpochs] = useState(24);
  const [augmentation, setAugmentation] = useState(true);
  const [fallbackLabel, setFallbackLabel] = useState<Digit>('ONE');
  const [samples, setSamples] = useState<TrainingSample[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [uploadState, setUploadState] = useState('Upload at least two samples for each digit.');
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const activeRun = training.data;
  const visibleSteps = useMemo(() => activeRun?.steps ?? [], [activeRun]);
  const counts = useMemo(() => DIGITS.map((label) => ({ label, count: samples.filter((sample) => sample.label === label).length })), [samples]);
  const canTrain = counts.every((item) => item.count >= 2);

  async function handleFiles(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    if (!selected.length) return;
    setUploadState('Reading labeled samples…');
    const next: TrainingSample[] = [];
    const names: string[] = [];
    try {
      for (const file of selected) {
        if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
          const sample = await extractMediaSample(file, fallbackLabel);
          if (sample) next.push(sample);
        } else {
          const parsed = parseStructuredText(await file.text(), file.name);
          if (!parsed.length) throw new Error(`${file.name}: no valid labeled 21-point samples found.`);
          next.push(...parsed);
        }
        names.push(file.name);
      }
      setSamples((current) => [...current, ...next].slice(0, 1000));
      setFiles((current) => [...current, ...names]);
      setUploadState(`${next.length} samples added. ${next.length ? 'Add more files or start training.' : ''}`);
    } catch (error) {
      setUploadState(error instanceof Error ? error.message : 'The selected files could not be read.');
    } finally {
      event.target.value = '';
    }
  }

  function removeSamplesFromFile(name: string) {
    setSamples((current) => current.filter((sample) => sample.source !== name));
    setFiles((current) => current.filter((file) => file !== name));
  }

  function startTraining() {
    if (!canTrain) return;
    training.mutate({ data: { epochs, augmentation, dataset: samples } }, {
      onSuccess: (run) => {
        setRuns((current) => [{ ...run, ranAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }, ...current].slice(0, 4));
        setUploadState('Training complete. The camera now uses this model.');
      },
    });
  }

  return <div className="min-h-[100dvh]">
    <PageHeader eyebrow="Training lab / labeled ISL data" title="Fit the instrument" description="Upload landmark files or labeled image/video examples. The server learns from your samples, holds out a deterministic validation split, and publishes the resulting model to the camera." action={<div className="flex items-center gap-2 border border-border bg-card px-3 py-2 font-mono text-[10px] uppercase tracking-[0.11em] text-muted-foreground"><Cpu size={14} className="text-primary" />Real dataset mode</div>} />
    <div className="mx-auto max-w-[1500px] space-y-5 px-5 py-6 sm:px-8 lg:space-y-6 lg:px-10 lg:py-8">
      <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <section className="panel-shadow border border-border bg-card p-5 sm:p-6 stagger-in" data-testid="section-training-controls">
          <div className="mb-6 flex items-start justify-between"><div><div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground"><SlidersHorizontal size={14} className="text-primary" />Dataset intake</div><p className="mt-1 text-xs text-muted-foreground">JSON, CSV, NDJSON, images, or video.</p></div><span className="font-mono text-[10px] text-muted-foreground">DATA / 01</span></div>
          <label className="group flex cursor-pointer items-center gap-3 border border-dashed border-primary/60 bg-primary/5 p-4 transition hover:bg-primary/10" data-testid="input-training-dataset">
            <UploadCloud size={20} className="text-primary" /><span className="flex-1"><b className="block text-xs">Choose labeled files</b><span className="mt-1 block text-[10px] leading-4 text-muted-foreground">Media labels come from ONE–FIVE in the filename, or use the fallback below.</span></span><FileUp size={15} className="text-muted-foreground" /><input type="file" multiple accept=".json,.jsonl,.csv,text/csv,image/*,video/*" onChange={handleFiles} className="sr-only" />
          </label>
          <div className="mt-3 flex items-center justify-between gap-3 text-[10px]"><span className="font-mono uppercase tracking-[0.12em] text-muted-foreground">Media fallback label</span><select value={fallbackLabel} onChange={(event) => setFallbackLabel(event.target.value as Digit)} className="border border-border bg-background px-2 py-1 text-xs">{DIGITS.map((label) => <option key={label}>{label}</option>)}</select></div>
          <div className="mt-3 border border-border bg-background p-3 text-[10px] leading-4 text-muted-foreground" data-testid="status-dataset-upload">{uploadState}</div>
          {files.length > 0 && <div className="mt-3 space-y-1">{files.map((file) => <div key={file} className="flex items-center gap-2 text-[10px]"><span className="min-w-0 flex-1 truncate">{file}</span><button type="button" onClick={() => removeSamplesFromFile(file)} className="text-muted-foreground hover:text-destructive" aria-label={`Remove ${file}`}><X size={12} /></button></div>)}</div>}
          <div className="mt-5 grid grid-cols-5 gap-1 border-t border-border pt-4">{counts.map((item) => <div key={item.label} className={`p-2 text-center ${item.count >= 2 ? 'bg-accent/40' : 'bg-muted'}`}><div className="font-mono text-[9px] text-muted-foreground">{item.label}</div><div className="mt-1 text-sm font-bold">{item.count}</div></div>)}</div>
          <div className="my-6 border-t border-border" />
          <label className="block"><span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Epoch budget</span><div className="mt-3 flex items-center gap-3"><input data-testid="input-training-epochs" type="range" min="1" max="80" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} className="h-1 flex-1 accent-[#f6c341]" /><span className="w-12 border border-border bg-background px-2 py-1 text-center font-mono text-xs font-bold">{epochs}</span></div></label>
          <label className="mt-6 flex cursor-pointer items-center justify-between gap-3" data-testid="toggle-augmentation"><span><span className="block text-xs font-bold">Geometry augmentation</span><span className="mt-1 block text-[10px] leading-4 text-muted-foreground">Use small landmark perturbations during fitting.</span></span><button type="button" role="switch" aria-checked={augmentation} onClick={() => setAugmentation((value) => !value)} className={`relative h-6 w-11 shrink-0 border transition-colors ${augmentation ? 'border-primary bg-primary' : 'border-border bg-muted'}`} data-testid="button-toggle-augmentation"><span className={`absolute top-1 size-4 bg-primary-foreground transition-transform ${augmentation ? 'translate-x-5' : 'translate-x-1'}`} /></button></label>
          <div className="mt-6 grid grid-cols-2 gap-3 border-t border-border pt-5"><div className="bg-background p-3"><Database size={15} className="mb-2 text-muted-foreground" /><div className="font-mono text-[9px] uppercase text-muted-foreground">samples</div><div className="mt-1 text-sm font-bold">{samples.length}</div></div><div className="bg-background p-3"><Gauge size={15} className="mb-2 text-muted-foreground" /><div className="font-mono text-[9px] uppercase text-muted-foreground">classes</div><div className="mt-1 text-sm font-bold">{counts.filter((item) => item.count > 0).length} / 5</div></div></div>
          <button onClick={startTraining} disabled={!canTrain || training.isPending} className="mt-6 flex w-full items-center justify-center gap-2 bg-primary px-4 py-3 text-xs font-bold text-primary-foreground transition hover:bg-[#f8d15f] disabled:cursor-not-allowed disabled:opacity-60" data-testid="button-run-training">{training.isPending ? <><span className="animate-pulse">Fitting model…</span><span className="size-3 animate-spin border-2 border-primary-foreground/30 border-t-primary-foreground" /></> : <><Play size={15} fill="currentColor" />Train on uploaded data</>}</button>
          {!canTrain && <div className="mt-3 text-[10px] text-muted-foreground">Need at least two samples in every digit class before training.</div>}
          {training.isError && <div className="mt-4 border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive" data-testid="status-training-error">Training service returned an error. Check that every sample has 21 landmarks.</div>}
        </section>
        <section className="space-y-4 stagger-in stagger-1" data-testid="section-training-trace">
          {activeRun && <RunSummary run={{ ...activeRun, ranAt: 'now' }} />}
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]"><div><LossChart steps={visibleSteps} /><div className="mt-2 flex items-center gap-4 font-mono text-[9px] uppercase tracking-[0.12em] text-muted-foreground"><span className="flex items-center gap-2"><span className="size-2 bg-[#178c83]" />loss</span><span className="flex items-center gap-2"><span className="size-2 bg-[#f6c341]" />epoch checkpoint</span></div></div><div className="border border-border bg-card p-5"><div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground"><LineChart size={14} className="text-primary" />Stage handoff</div><div className="mt-7 space-y-5"><div><div className="flex justify-between text-xs font-bold"><span>ANFIS</span><span className="font-mono text-muted-foreground">{activeRun ? 'fit to data' : 'waiting'}</span></div><div className="mt-2 h-1.5 bg-muted"><div className={`h-full bg-[#178c83] transition-all duration-700 ${activeRun ? 'w-full' : 'w-0'}`} /></div></div><div className="ml-5 border-l border-dashed border-border pl-4"><div className="text-[10px] leading-4 text-muted-foreground">Per-digit feature distributions are learned from the uploaded landmarks.</div></div><div><div className="flex justify-between text-xs font-bold"><span>GRU</span><span className="font-mono text-muted-foreground">{activeRun ? 'validated' : 'queued'}</span></div><div className="mt-2 h-1.5 bg-muted"><div className={`h-full bg-primary transition-all duration-700 ${activeRun ? 'w-full' : 'w-0'}`} /></div></div></div></div></div>
        </section>
      </div>
      <section className="border border-border bg-card stagger-in stagger-2" data-testid="section-run-history">
        <div className="flex items-center justify-between border-b border-border px-5 py-4"><div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground"><RotateCcw size={14} className="text-primary" />Session run history</div><span className="font-mono text-[10px] text-muted-foreground">{runs.length} measured runs</span></div>
        {runs.length === 0 ? <div className="flex items-center gap-3 px-5 py-8 text-xs text-muted-foreground"><Sparkles size={16} className="text-primary" /><span>Upload a balanced labeled dataset to create the first measured run.</span></div> : <div className="divide-y divide-border">{runs.map((run, index) => <div key={`${run.ranAt}-${index}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-4 px-5 py-4 text-xs" data-testid={`row-training-run-${index}`}><div><div className="font-bold">Trained recognizer / {run.datasetSamples} samples</div><div className="mt-1 font-mono text-[9px] uppercase text-muted-foreground">{run.ranAt} · {run.status}</div></div><span className="font-mono text-muted-foreground">{((run.validationAccuracy ?? 0) * 100).toFixed(1)}% valid.</span><span className="flex items-center gap-1 text-[#178c83]"><Check size={13} /> active</span></div>)}</div>}
      </section>
    </div>
  </div>;
}