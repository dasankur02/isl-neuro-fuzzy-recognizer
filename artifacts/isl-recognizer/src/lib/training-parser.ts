export const DIGITS = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'] as const;
export type Digit = (typeof DIGITS)[number];

export type Landmark = { x: number; y: number; z: number };
export type TrainingSample = {
  label: Digit;
  landmarks: Landmark[];
  sequence?: Landmark[][];
  source?: string;
};

type VisionResult = { landmarks?: Landmark[][] };
export type VisionDetector = {
  detect: (image: CanvasImageSource) => VisionResult;
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => VisionResult;
  close: () => void;
};
type VisionModule = {
  FilesetResolver: { forVisionTasks: (wasmPath: string) => Promise<unknown> };
  HandLandmarker: {
    createFromOptions: (
      resolver: unknown,
      options: Record<string, unknown>,
    ) => Promise<VisionDetector>;
  };
};

const DIGIT_ALIASES: Record<string, Digit> = {
  ONE: 'ONE',
  1: 'ONE',
  TWO: 'TWO',
  2: 'TWO',
  THREE: 'THREE',
  3: 'THREE',
  FOUR: 'FOUR',
  4: 'FOUR',
  FIVE: 'FIVE',
  5: 'FIVE',
};

const VISION_BUNDLE_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
const VISION_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export function labelFromValue(value: unknown): Digit | undefined {
  return DIGIT_ALIASES[String(value ?? '').trim().toUpperCase()];
}

export function asLandmarks(value: unknown): Landmark[] | undefined {
  if (!Array.isArray(value) || value.length !== 21) return undefined;
  const points = value.map((point) => {
    if (!point || typeof point !== 'object') return undefined;
    const item = point as Record<string, unknown>;
    return { x: Number(item.x), y: Number(item.y), z: Number(item.z ?? 0) };
  });
  return points.every(
    (point) =>
      point && [point.x, point.y, point.z].every(Number.isFinite),
  )
    ? (points as Landmark[])
    : undefined;
}

function sampleFromObject(
  value: unknown,
  source: string,
  fallbackLabel?: Digit,
): TrainingSample | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const item = value as Record<string, unknown>;
  const label = labelFromValue(item.label) ?? fallbackLabel;
  const landmarks = asLandmarks(item.landmarks);
  const sequence = Array.isArray(item.sequence)
    ? item.sequence
        .map(asLandmarks)
        .filter((frame): frame is Landmark[] => Boolean(frame))
    : undefined;
  const selected = landmarks ?? sequence?.at(-1);
  if (!label || !selected) return undefined;
  return {
    label,
    landmarks: selected,
    sequence: sequence?.length ? sequence : undefined,
    source,
  };
}

export function parseStructuredText(
  text: string,
  source: string,
): TrainingSample[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  try {
    const parsed: unknown = source.toLowerCase().endsWith('.jsonl')
      ? trimmed
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => JSON.parse(line))
      : JSON.parse(trimmed);
    const values = Array.isArray(parsed)
      ? parsed
      : parsed &&
          typeof parsed === 'object' &&
          Array.isArray((parsed as Record<string, unknown>).samples)
        ? (parsed as Record<string, unknown>).samples as unknown[]
        : [parsed];
    const samples = values.flatMap((value) => {
      const direct = sampleFromObject(value, source);
      if (direct) return [direct];
      if (value && typeof value === 'object') {
        return Object.entries(value as Record<string, unknown>).flatMap(
          ([key, entries]) => {
            const label = labelFromValue(key);
            if (!label || !Array.isArray(entries)) return [];
            return entries
              .map((entry) =>
                sampleFromObject({ label, landmarks: entry }, source),
              )
              .filter(
                (sample): sample is TrainingSample => Boolean(sample),
              );
          },
        );
      }
      return [];
    });
    if (samples.length) return samples;
  } catch {
    // CSV is handled below, and the caller reports invalid files if neither parser succeeds.
  }

  const rows = trimmed
    .split(/\r?\n/)
    .map((row) => row.split(',').map((cell) => cell.trim()));
  const first = rows[0] ?? [];
  const start = labelFromValue(first[0]) ? 0 : 1;
  return rows.slice(start).flatMap((row) => {
    const label = labelFromValue(row[0]);
    const numbers = row.slice(1).map(Number);
    if (
      !label ||
      numbers.length < 63 ||
      numbers.slice(0, 63).some((number) => !Number.isFinite(number))
    )
      return [];
    const landmarks = Array.from({ length: 21 }, (_, index) => ({
      x: numbers[index * 3],
      y: numbers[index * 3 + 1],
      z: numbers[index * 3 + 2],
    }));
    return [{ label, landmarks, source }];
  });
}

async function createDetector(
  runningMode: 'IMAGE' | 'VIDEO',
): Promise<VisionDetector> {
  const vision = (await import(
    /* @vite-ignore */ VISION_BUNDLE_URL
  )) as unknown as VisionModule;
  const resolver = await vision.FilesetResolver.forVisionTasks(VISION_WASM_URL);
  return vision.HandLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: 'GPU' },
    runningMode,
    numHands: 1,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
  });
}

function mediaLabel(file: File, fallback?: Digit) {
  const match = file.name.match(
    /(?:^|[^a-z])(one|two|three|four|five|[1-5])(?:[^a-z]|$)/i,
  );
  return labelFromValue(match?.[1]) ?? fallback;
}

export async function extractMediaSample(
  file: File,
  fallback?: Digit,
  detectorFactory: (
    runningMode: 'IMAGE' | 'VIDEO',
  ) => Promise<VisionDetector> = createDetector,
): Promise<TrainingSample | undefined> {
  const label = mediaLabel(file, fallback);
  if (!label)
    throw new Error(
      `${file.name}: choose a label or include ONE–FIVE in the filename.`,
    );
  const detector = await detectorFactory(
    file.type.startsWith('image/') ? 'IMAGE' : 'VIDEO',
  );
  try {
    if (file.type.startsWith('image/')) {
      const bitmap = await createImageBitmap(file);
      try {
        const landmarks = detector.detect(bitmap).landmarks?.[0];
        if (!landmarks) return undefined;
        return { label, landmarks, source: file.name };
      } finally {
        bitmap.close();
      }
    }
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () =>
        reject(new Error(`${file.name}: video could not be read.`));
    });
    const frameCount = Math.max(
      1,
      Math.min(
        60,
        Math.ceil(
          (Number.isFinite(video.duration) ? video.duration : 1) * 4,
        ),
      ),
    );
    const sequence: Landmark[][] = [];
    for (let index = 0; index < frameCount; index += 1) {
      video.currentTime = Number.isFinite(video.duration)
        ? (video.duration * index) / Math.max(frameCount - 1, 1)
        : 0;
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
      });
      const landmarks = detector.detectForVideo(video, performance.now())
        .landmarks?.[0];
      if (landmarks) sequence.push(landmarks);
    }
    URL.revokeObjectURL(url);
    const landmarks = sequence.at(-1);
    return landmarks
      ? { label, landmarks, sequence, source: file.name }
      : undefined;
  } finally {
    detector.close();
  }
}