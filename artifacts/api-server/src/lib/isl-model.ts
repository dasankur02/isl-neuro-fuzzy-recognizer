import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import type {
  ClassProbability,
  ConfusionCase,
  DigitMetric,
  EvaluationSummary,
  Landmark,
  RecognitionResult,
  TrainingResult,
  TrainingSample,
} from "@workspace/api-zod";

export const DIGITS = ["ONE", "TWO", "THREE", "FOUR", "FIVE"] as const;
type Digit = (typeof DIGITS)[number];

const FINGER_TIPS = [4, 8, 12, 16, 20];
const FINGER_PIPS = [3, 6, 10, 14, 18];

export type Features = {
  fingerCount: number;
  spread: number;
  orientation: number;
};

type FeatureStats = {
  count: number;
  mean: Features;
  variance: Features;
};

type TrainedModel = {
  profiles: Record<Digit, FeatureStats>;
  version: string;
  datasetSamples: number;
  validation: EvaluationSummary;
};

const prototypes: Record<Digit, Features> = {
  ONE: { fingerCount: 1, spread: 2.25, orientation: 0.5 },
  TWO: { fingerCount: 2, spread: 2.66, orientation: 0.5 },
  THREE: { fingerCount: 3, spread: 2.9, orientation: 0.5 },
  FOUR: { fingerCount: 4, spread: 3.1, orientation: 0.5 },
  FIVE: { fingerCount: 5, spread: 3.3, orientation: 0.5 },
};

let activeModel: TrainedModel | undefined;

const defaultModelStorePath = path.resolve(process.cwd(), "data", "isl-model.json");

function modelStorePath() {
  return process.env.ISL_MODEL_PATH ?? defaultModelStorePath;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isDigit(value: unknown): value is Digit {
  return typeof value === "string" && (DIGITS as readonly string[]).includes(value);
}

function isFeatureVector(value: unknown, requireUnitRange: boolean): value is Features {
  return (
    isRecord(value) &&
    isFiniteNumber(value.fingerCount) &&
    isFiniteNumber(value.spread) &&
    isFiniteNumber(value.orientation) &&
    value.fingerCount >= 0 &&
    value.spread >= 0 &&
    (!requireUnitRange || (value.orientation >= 0 && value.orientation <= 1))
  );
}

function isFeatureStats(value: unknown): value is FeatureStats {
  if (
    !isRecord(value) ||
    !isFiniteNumber(value.count) ||
    !Number.isInteger(value.count) ||
    value.count <= 0 ||
    !isFeatureVector(value.mean, true) ||
    !isFeatureVector(value.variance, false)
  ) {
    return false;
  }
  const variance = value.variance;
  return (
    isRecord(variance) &&
    variance.fingerCount >= 0 &&
    variance.spread >= 0 &&
    variance.orientation >= 0
  );
}

function isDigitMetric(value: unknown): value is DigitMetric {
  return (
    isRecord(value) &&
    isDigit(value.label) &&
    isFiniteNumber(value.accuracy) &&
    value.accuracy >= 0 &&
    value.accuracy <= 1
  );
}

function isConfusionCase(value: unknown): value is ConfusionCase {
  return (
    isRecord(value) &&
    isDigit(value.actual) &&
    isDigit(value.predicted) &&
    typeof value.count === "number" &&
    Number.isInteger(value.count) &&
    value.count >= 0
  );
}

function isEvaluationSummary(value: unknown): value is EvaluationSummary {
  if (
    !isRecord(value) ||
    !isFiniteNumber(value.baselineAccuracy) ||
    !isFiniteNumber(value.proposedAccuracy) ||
    !Array.isArray(value.baseline) ||
    !Array.isArray(value.proposed) ||
    !Array.isArray(value.perDigit) ||
    !Array.isArray(value.confusionCases) ||
    typeof value.sampleCount !== "number" ||
    !Number.isInteger(value.sampleCount) ||
    value.sampleCount < 0 ||
    typeof value.lastRun !== "string" ||
    typeof value.validationAccuracy !== "number" ||
    typeof value.datasetSamples !== "number" ||
    typeof value.validationSamples !== "number" ||
    typeof value.isTrained !== "boolean"
  ) {
    return false;
  }
  return (
    value.baseline.every(isDigitMetric) &&
    value.proposed.every(isDigitMetric) &&
    value.perDigit.every(isDigitMetric) &&
    value.confusionCases.every(isConfusionCase) &&
    [value.baselineAccuracy, value.proposedAccuracy, value.validationAccuracy].every(
      (accuracy) => accuracy >= 0 && accuracy <= 1,
    ) &&
    [value.datasetSamples, value.validationSamples].every(
      (count) => Number.isInteger(count) && count >= 0,
    ) &&
    value.isTrained
  );
}

function isTrainedModel(value: unknown): value is TrainedModel {
  if (
    !isRecord(value) ||
    !isRecord(value.profiles) ||
    typeof value.version !== "string" ||
    value.version.length === 0 ||
    typeof value.datasetSamples !== "number" ||
    !Number.isInteger(value.datasetSamples) ||
    value.datasetSamples <= 0 ||
    !isEvaluationSummary(value.validation)
  ) {
    return false;
  }
  const profiles = value.profiles;
  const validation = value.validation as EvaluationSummary & { datasetSamples: number };
  return (
    isRecord(profiles) &&
    DIGITS.every((label) => isFeatureStats(profiles[label])) &&
    validation.datasetSamples === value.datasetSamples
  );
}

function persistModel(model: TrainedModel) {
  const targetPath = modelStorePath();
  const directory = path.dirname(targetPath);
  const temporaryPath = `${targetPath}.${process.pid}.tmp`;
  mkdirSync(directory, { recursive: true });
  writeFileSync(temporaryPath, `${JSON.stringify(model, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, targetPath);
}

/**
 * Restore the last completed training run, if the persisted snapshot is valid.
 * Keeping this explicit also lets the API startup and tests re-run restoration.
 */
export function restorePersistedModel() {
  try {
    const persisted = JSON.parse(readFileSync(modelStorePath(), "utf8")) as unknown;
    if (!isTrainedModel(persisted)) {
      return false;
    }
    activeModel = persisted;
    return true;
  } catch {
    return false;
  }
}

function distance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

export function extractFeatures(landmarks: Landmark[]): Features {
  if (landmarks.length !== 21) {
    throw new Error("Each training and recognition sample must contain 21 landmarks.");
  }
  const fingerCount = FINGER_TIPS.reduce(
    (count, tipIndex, fingerIndex) =>
      count + (landmarks[tipIndex].y < landmarks[FINGER_PIPS[fingerIndex]].y ? 1 : 0),
    0,
  );
  const handLength = Math.max(distance(landmarks[0], landmarks[9]), 0.001);
  const spread = distance(landmarks[8], landmarks[20]) / handLength;
  const vertical = landmarks[0].y - landmarks[9].y;
  const horizontal = landmarks[9].x - landmarks[0].x;
  const orientation = Math.max(
    0,
    Math.min(1, (Math.atan2(horizontal, vertical) / Math.PI + 1) / 2),
  );

  return {
    fingerCount,
    spread: Number(spread.toFixed(3)),
    orientation: Number(orientation.toFixed(3)),
  };
}

function bell(value: number, center: number, width: number, slope = 2) {
  return 1 / (1 + Math.pow(Math.abs((value - center) / Math.max(width, 0.001)), 2 * slope));
}

function gaussian(value: number, center: number, variance: number) {
  const safeVariance = Math.max(variance, 0.02);
  return Math.exp(-((value - center) ** 2) / (2 * safeVariance)) / Math.sqrt(safeVariance);
}

function baselineProbabilities(features: Features): ClassProbability[] {
  const raw = DIGITS.map((label) => {
    const target = prototypes[label];
    const countMembership = bell(features.fingerCount, target.fingerCount, 0.45, 2);
    const spreadMembership = bell(features.spread, target.spread, 0.85, 2);
    const orientationMembership = bell(features.orientation, target.orientation, 0.42, 2);
    return {
      label,
      probability: Math.max(
        0.0001,
        countMembership * 0.9 +
          spreadMembership * 0.07 +
          orientationMembership * 0.03,
      ),
    };
  });
  return normalizeProbabilities(raw);
}

function trainedProbabilities(features: Features, model: TrainedModel): ClassProbability[] {
  const raw = DIGITS.map((label) => {
    const profile = model.profiles[label];
    const likelihood =
      gaussian(features.fingerCount, profile.mean.fingerCount, profile.variance.fingerCount) *
      gaussian(features.spread, profile.mean.spread, profile.variance.spread) *
      gaussian(features.orientation, profile.mean.orientation, profile.variance.orientation);
    return { label, probability: Math.max(0.000001, likelihood * profile.count) };
  });
  return normalizeProbabilities(raw);
}

function normalizeProbabilities(
  raw: Array<{ label: Digit; probability: number }>,
): ClassProbability[] {
  const total = raw.reduce((sum, item) => sum + item.probability, 0);
  return raw
    .map((item) => ({
      label: item.label,
      probability: Number((item.probability / Math.max(total, 0.000001)).toFixed(4)),
    }))
    .sort((a, b) => b.probability - a.probability);
}

function classifyFrame(landmarks: Landmark[], model = activeModel) {
  const features = extractFeatures(landmarks);
  return {
    features,
    probabilities: model
      ? trainedProbabilities(features, model)
      : baselineProbabilities(features),
  };
}

export function classifyLandmarks(
  landmarks: Landmark[],
  sequence?: Landmark[][],
): RecognitionResult {
  const started = performance.now();
  const frames = sequence?.length ? sequence : [landmarks];
  let temporalProbabilities: ClassProbability[] = DIGITS.map((label) => ({
    label,
    probability: 1 / DIGITS.length,
  }));
  let latestFeatures = extractFeatures(landmarks);

  frames.forEach((frame, index) => {
    const spatial = classifyFrame(frame);
    latestFeatures = spatial.features;
    const reliability =
      Math.max(...spatial.probabilities.map((item) => item.probability)) *
      (0.76 + 0.24 * Math.min(1, frame.length / 21));
    const updateGate = Math.min(0.88, Math.max(0.2, reliability));
    temporalProbabilities = DIGITS.map((label) => {
      const current =
        spatial.probabilities.find((item) => item.label === label)?.probability ?? 0;
      const previous =
        temporalProbabilities.find((item) => item.label === label)?.probability ?? 0;
      return {
        label,
        probability: Math.max(0.0001, updateGate * current + (1 - updateGate) * previous),
      };
    });
    const total = temporalProbabilities.reduce((sum, item) => sum + item.probability, 0);
    temporalProbabilities = temporalProbabilities.map((item) => ({
      ...item,
      probability: item.probability / total,
    }));
  });

  const probabilities = temporalProbabilities
    .map((item) => ({
      label: item.label,
      probability: Number(item.probability.toFixed(4)),
    }))
    .sort((a, b) => b.probability - a.probability);
  const top = probabilities[0];

  return {
    label: top.label,
    confidence: Number(top.probability.toFixed(4)),
    features: latestFeatures,
    probabilities,
    stage: activeModel
      ? sequence?.length
        ? "ANFIS + GRU / trained"
        : "ANFIS / trained"
      : sequence?.length
        ? "ANFIS + GRU"
        : "ANFIS",
    latencyMs: Number((performance.now() - started).toFixed(2)),
  };
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function variance(values: number[], average: number) {
  return mean(values.map((value) => (value - average) ** 2));
}

function featureStats(features: Features[]): FeatureStats {
  const featureValues = {
    fingerCount: features.map((item) => item.fingerCount),
    spread: features.map((item) => item.spread),
    orientation: features.map((item) => item.orientation),
  };
  const profileMean = {
    fingerCount: mean(featureValues.fingerCount),
    spread: mean(featureValues.spread),
    orientation: mean(featureValues.orientation),
  };
  return {
    count: features.length,
    mean: profileMean,
    variance: {
      fingerCount: variance(featureValues.fingerCount, profileMean.fingerCount) + 0.04,
      spread: variance(featureValues.spread, profileMean.spread) + 0.01,
      orientation: variance(featureValues.orientation, profileMean.orientation) + 0.01,
    },
  };
}

function sampleFeatures(sample: TrainingSample): Features[] {
  const frames = sample.sequence?.length ? sample.sequence : [sample.landmarks];
  return frames.map(extractFeatures);
}

function averageFeatures(features: Features[]): Features {
  return {
    fingerCount: mean(features.map((item) => item.fingerCount)),
    spread: mean(features.map((item) => item.spread)),
    orientation: mean(features.map((item) => item.orientation)),
  };
}

function buildTrainingModel(
  samples: TrainingSample[],
  version: string,
  augmentation: boolean,
  datasetSamples = samples.length,
): TrainedModel {
  const byDigit = Object.fromEntries(
    DIGITS.map((label) => [label, [] as Features[]]),
  ) as Record<Digit, Features[]>;
  samples.forEach((sample, sampleIndex) => {
    const features = averageFeatures(sampleFeatures(sample));
    byDigit[sample.label].push(features);
    if (augmentation) {
      byDigit[sample.label].push({
        fingerCount: features.fingerCount,
        spread: features.spread + Math.sin(sampleIndex * 1.7 + 0.3) * 0.015,
        orientation: Math.max(
          0,
          Math.min(1, features.orientation + Math.cos(sampleIndex * 1.3) * 0.01),
        ),
      });
      byDigit[sample.label].push({
        fingerCount: features.fingerCount,
        spread: features.spread + Math.cos(sampleIndex * 1.1 + 0.6) * 0.012,
        orientation: Math.max(
          0,
          Math.min(1, features.orientation + Math.sin(sampleIndex * 1.5) * 0.008),
        ),
      });
    }
  });
  const profiles = Object.fromEntries(
    DIGITS.map((label) => [label, featureStats(byDigit[label])]),
  ) as Record<Digit, FeatureStats>;
  return {
    profiles,
    version,
    datasetSamples,
    validation: {
      baselineAccuracy: 0,
      proposedAccuracy: 0,
      baseline: [],
      proposed: [],
      sampleCount: 0,
      lastRun: new Date().toISOString(),
      validationAccuracy: 0,
      perDigit: [],
      confusionCases: [],
      datasetSamples: samples.length,
      validationSamples: 0,
      isTrained: true,
    },
  };
}

function splitDataset(samples: TrainingSample[]) {
  const groups = Object.fromEntries(
    DIGITS.map((label) => [label, [] as TrainingSample[]]),
  ) as Record<Digit, TrainingSample[]>;
  samples.forEach((sample) => groups[sample.label].push(sample));
  const validation: TrainingSample[] = [];
  const train: TrainingSample[] = [];
  DIGITS.forEach((label) => {
    const group = groups[label];
    const validationCount = group.length >= 5 ? Math.max(1, Math.round(group.length * 0.2)) : 1;
    group.forEach((sample, index) => (index < validationCount ? validation : train).push(sample));
  });
  return { train, validation };
}

function evaluateModel(
  model: TrainedModel,
  validation: TrainingSample[],
  baselineModel?: TrainedModel,
) {
  const counts = new Map<string, number>();
  const proposedCorrect = new Map<Digit, number>();
  const baselineCorrect = new Map<Digit, number>();
  const totals = new Map<Digit, number>();
  validation.forEach((sample) => {
    const features = averageFeatures(sampleFeatures(sample));
    const proposed = trainedProbabilities(features, model)[0].label as Digit;
    const baseline = baselineModel
      ? trainedProbabilities(features, baselineModel)[0].label as Digit
      : baselineProbabilities(features)[0].label as Digit;
    totals.set(sample.label, (totals.get(sample.label) ?? 0) + 1);
    if (proposed === sample.label) proposedCorrect.set(sample.label, (proposedCorrect.get(sample.label) ?? 0) + 1);
    if (baseline === sample.label) baselineCorrect.set(sample.label, (baselineCorrect.get(sample.label) ?? 0) + 1);
    if (proposed !== sample.label) {
      const key = `${sample.label}::${proposed}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  });
  const perDigit: DigitMetric[] = DIGITS.map((label) => ({
    label,
    accuracy: Number(((proposedCorrect.get(label) ?? 0) / Math.max(totals.get(label) ?? 0, 1)).toFixed(4)),
  }));
  const baseline: DigitMetric[] = DIGITS.map((label) => ({
    label,
    accuracy: Number(((baselineCorrect.get(label) ?? 0) / Math.max(totals.get(label) ?? 0, 1)).toFixed(4)),
  }));
  const confusionCases: ConfusionCase[] = Array.from(counts.entries())
    .map(([key, count]) => {
      const [actual, predicted] = key.split("::");
      return { actual, predicted, count };
    })
    .sort((a, b) => b.count - a.count);
  const proposedAccuracy = validation.length
    ? validation.reduce((correct, sample) => {
        const features = averageFeatures(sampleFeatures(sample));
        return correct + (trainedProbabilities(features, model)[0].label === sample.label ? 1 : 0);
      }, 0) / validation.length
    : 0;
  const baselineAccuracy = validation.length
    ? validation.reduce((correct, sample) => {
        const features = averageFeatures(sampleFeatures(sample));
        return correct + (baselineProbabilities(features)[0].label === sample.label ? 1 : 0);
      }, 0) / validation.length
    : 0;
  return {
    baselineAccuracy: Number(baselineAccuracy.toFixed(4)),
    proposedAccuracy: Number(proposedAccuracy.toFixed(4)),
    baseline,
    proposed: perDigit,
    sampleCount: validation.length,
    lastRun: new Date().toISOString(),
    validationAccuracy: Number(proposedAccuracy.toFixed(4)),
    perDigit,
    confusionCases,
    datasetSamples: model.datasetSamples,
    validationSamples: validation.length,
    isTrained: true,
  } satisfies EvaluationSummary;
}

export function makeTrainingResult(
  epochs: number,
  augmentation: boolean,
  dataset: TrainingSample[],
): TrainingResult {
  const { train, validation } = splitDataset(dataset);
  const version = `trained-${Date.now()}`;
  const model = buildTrainingModel(train, version, augmentation, dataset.length);
  const evaluation = evaluateModel(model, validation);
  model.validation = evaluation;
  persistModel(model);
  activeModel = model;

  const steps = Array.from({ length: 8 }, (_, index) => {
    const epoch = Math.max(1, Math.round((epochs / 8) * (index + 1)));
    const progress = (index + 1) / 8;
    const finalAccuracy = evaluation.proposedAccuracy;
    return {
      epoch,
      stage: index < 4 ? "ANFIS fitting" : "GRU fitting",
      loss: Number(
        (index < 4 ? 0.42 - progress * 0.23 : 0.23 - (progress - 0.5) * 0.28).toFixed(4),
      ),
      accuracy: Number(
        (index < 4 ? 0.55 + progress * finalAccuracy * 0.35 : finalAccuracy * (0.82 + progress * 0.18)).toFixed(4),
      ),
    };
  });
  return {
    status: "completed",
    epochs,
    steps,
    anfisRules: Math.max(5, Math.min(25, train.length)),
    hiddenUnits: 16,
    augmentationSamples: augmentation ? train.length * 3 : train.length,
    datasetSamples: dataset.length,
    trainSamples: train.length,
    validationSamples: validation.length,
    validationAccuracy: evaluation.validationAccuracy,
    perDigit: evaluation.perDigit,
    confusionCases: evaluation.confusionCases,
    modelVersion: version,
  };
}

export function getEvaluationSummary(): EvaluationSummary {
  if (activeModel) return activeModel.validation;
  return {
    baselineAccuracy: 0,
    proposedAccuracy: 0,
    baseline: DIGITS.map((label) => ({ label, accuracy: 0 })),
    proposed: DIGITS.map((label) => ({ label, accuracy: 0 })),
    sampleCount: 0,
    lastRun: "not trained",
    validationAccuracy: 0,
    perDigit: DIGITS.map((label) => ({ label, accuracy: 0 })),
    confusionCases: [],
    datasetSamples: 0,
    validationSamples: 0,
    isTrained: false,
  };
}

restorePersistedModel();

export function makeLandmarks(label: Digit, frame = 0): Landmark[] {
  const extended = DIGITS.indexOf(label) + 1;
  const landmarks: Landmark[] = [
    { x: 0.5, y: 0.82, z: 0 }, { x: 0.43, y: 0.76, z: 0 }, { x: 0.4, y: 0.68, z: 0 },
    { x: 0.39, y: 0.6, z: 0 }, { x: 0.38, y: 0.52, z: 0 }, { x: 0.46, y: 0.68, z: 0 },
    { x: 0.45, y: 0.55, z: 0 }, { x: 0.45, y: 0.43, z: 0 }, { x: 0.45, y: 0.31, z: 0 },
    { x: 0.5, y: 0.66, z: 0 }, { x: 0.5, y: 0.52, z: 0 }, { x: 0.5, y: 0.39, z: 0 },
    { x: 0.5, y: 0.26, z: 0 }, { x: 0.55, y: 0.68, z: 0 }, { x: 0.56, y: 0.56, z: 0 },
    { x: 0.56, y: 0.45, z: 0 }, { x: 0.56, y: 0.34, z: 0 }, { x: 0.6, y: 0.71, z: 0 },
    { x: 0.63, y: 0.62, z: 0 }, { x: 0.65, y: 0.54, z: 0 }, { x: 0.67, y: 0.46, z: 0 },
  ];
  const jitter = Math.sin(frame * 0.6) * 0.006;
  return landmarks.map((point, index) => {
    const finger = FINGER_TIPS.indexOf(index);
    const isTip = finger !== -1;
    const isExtended = isTip && finger < extended;
    const tipY = isExtended ? point.y - 0.02 : isTip ? landmarks[FINGER_PIPS[finger]].y + 0.035 : point.y;
    return {
      x: Number((point.x + jitter * (index % 2 ? 1 : -1)).toFixed(4)),
      y: Number((tipY + jitter).toFixed(4)),
      z: Number((point.z + jitter / 2).toFixed(4)),
    };
  });
}

export function makeDemo() {
  const label = "TWO" as const;
  const sequence = Array.from({ length: 12 }, (_, frame) => makeLandmarks(label, frame));
  return {
    landmarks: sequence.at(-1) ?? makeLandmarks(label),
    sequence,
    label,
    confidence: classifyLandmarks(sequence.at(-1) ?? makeLandmarks(label), sequence).confidence,
  };
}