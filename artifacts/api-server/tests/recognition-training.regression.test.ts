import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { after, before, describe, it } from 'node:test';
import express from 'express';
import type { TrainingSample } from '@workspace/api-zod';
import {
  classifyLandmarks,
  getEvaluationSummary,
  makeLandmarks,
  makeTrainingResult,
} from '../src/lib/isl-model.ts';
import recognitionRouter from '../src/routes/recognition.ts';
import trainingRouter from '../src/routes/training.ts';
import {
  extractMediaSample,
  parseStructuredText,
  type Landmark,
  type VisionDetector,
} from '../../isl-recognizer/src/lib/training-parser.ts';

const labels = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'] as const;

function dataset(samplesPerDigit = 2): TrainingSample[] {
  return labels.flatMap((label) =>
    Array.from({ length: samplesPerDigit }, (_, frame) => ({
      label,
      landmarks: makeLandmarks(label, frame),
      source: `${label.toLowerCase()}-${frame}.json`,
    })),
  );
}

function landmarkRow(offset = 0): number[] {
  return Array.from({ length: 21 * 3 }, (_, index) => index + offset);
}

describe('uploaded dataset training regressions', () => {
  let server: ReturnType<typeof createServer>;
  let baseUrl: string;

  before(async () => {
    const testApp = express();
    testApp.use(express.json());
    testApp.use('/api', trainingRouter);
    testApp.use('/api', recognitionRouter);
    server = createServer(testApp);
    await new Promise<void>((resolve) => {
      server.listen(0, '127.0.0.1', () => resolve());
    });
    const address = server.address();
    assert(address && typeof address !== 'string');
    baseUrl = `http://127.0.0.1:${address.port}/api`;
  });

  after(async () => {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  });

  it('accepts a balanced dataset and reports held-out metrics', async () => {
    const response = await fetch(`${baseUrl}/training/run`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ epochs: 24, augmentation: false, dataset: dataset() }),
    });
    assert.equal(response.status, 200);
    const result = await response.json();

    assert.equal(result.status, 'completed');
    assert.equal(result.datasetSamples, 10);
    assert.equal(result.trainSamples, 5);
    assert.equal(result.validationSamples, 5);
    assert.equal(result.perDigit.length, 5);
    assert.deepEqual(
      result.perDigit.map((metric: { label: string }) => metric.label),
      labels,
    );
    assert.equal(typeof result.validationAccuracy, 'number');
    assert.ok(Array.isArray(result.confusionCases));

    const evaluation = getEvaluationSummary();
    assert.equal(evaluation.isTrained, true);
    assert.equal(evaluation.datasetSamples, 10);
    assert.equal(evaluation.validationSamples, 5);

    const recognitionResponse = await fetch(`${baseUrl}/recognition/classify`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ landmarks: makeLandmarks('FIVE') }),
    });
    assert.equal(recognitionResponse.status, 200);
    assert.equal((await recognitionResponse.json()).stage, 'ANFIS / trained');
  });

  it('rejects datasets that do not contain every digit twice', async () => {
    const response = await fetch(`${baseUrl}/training/run`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        dataset: dataset().filter((sample) => sample.label !== 'FIVE'),
      }),
    });
    assert.equal(response.status, 400);
    assert.match((await response.json()).error, /two labeled landmark samples/i);
  });

  it('rejects malformed landmark arrays before training', async () => {
    const invalid = dataset().map((sample) => ({
      ...sample,
      landmarks: sample.landmarks.slice(0, 20),
    }));
    const response = await fetch(`${baseUrl}/training/run`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ dataset: invalid }),
    });
    assert.equal(response.status, 400);
    assert.match((await response.json()).error, /invalid|upload/i);
  });

  it('keeps the recognition output on the trained stage after training', async () => {
    const result = makeTrainingResult(12, true, dataset());
    assert.equal(result.status, 'completed');
    const recognition = classifyLandmarks(makeLandmarks('FIVE'));
    assert.equal(recognition.stage, 'ANFIS / trained');
    assert.ok(recognition.probabilities.length === 5);
  });
});

describe('uploaded dataset parsers', () => {
  const landmarks = (offset = 0): Landmark[] =>
    Array.from({ length: 21 }, (_, index) => ({
      x: index + offset,
      y: index + offset + 0.1,
      z: index + offset + 0.2,
    }));

  it('parses JSON arrays and wrapper objects', () => {
    const json = JSON.stringify({
      samples: [
        { label: 'ONE', landmarks: landmarks() },
        { label: 2, landmarks: landmarks(10) },
      ],
    });
    const samples = parseStructuredText(json, 'hands.json');
    assert.equal(samples.length, 2);
    assert.deepEqual(
      samples.map((sample) => sample.label),
      ['ONE', 'TWO'],
    );
  });

  it('parses CSV rows with and without a header', () => {
    const values = landmarkRow().join(',');
    const samples = parseStructuredText(`label,${values}\nTHREE,${values}`, 'hands.csv');
    assert.equal(samples.length, 1);
    assert.equal(samples[0].label, 'THREE');
    assert.equal(samples[0].landmarks.length, 21);
    assert.equal(parseStructuredText(`FOUR,${values}`, 'hands.csv')[0].label, 'FOUR');
  });

  it('parses one JSON object per NDJSON line and skips malformed landmarks', () => {
    const text = [
      JSON.stringify({ label: 'ONE', landmarks: landmarks() }),
      JSON.stringify({ label: 'TWO', landmarks: landmarks(1).slice(0, 20) }),
      JSON.stringify({ label: 'THREE', landmarks: landmarks(2) }),
    ].join('\n');
    const samples = parseStructuredText(text, 'hands.jsonl');
    assert.equal(samples.length, 2);
    assert.deepEqual(
      samples.map((sample) => sample.label),
      ['ONE', 'THREE'],
    );
  });

  it('reports an image without detected landmarks as an extraction failure', async () => {
    const originalBitmap = (globalThis as Record<string, unknown>).createImageBitmap;
    (globalThis as Record<string, unknown>).createImageBitmap = async () => ({
      close() {},
    });
    let closed = false;
    const detector = {
      detect: () => ({}),
      detectForVideo: () => ({}),
      close: () => {
        closed = true;
      },
    } as VisionDetector;
    try {
      const sample = await extractMediaSample(
        new File(['image'], 'ONE-image.png', { type: 'image/png' }),
        undefined,
        async () => detector,
      );
      assert.equal(sample, undefined);
      assert.equal(closed, true);
    } finally {
      if (originalBitmap) {
        (globalThis as Record<string, unknown>).createImageBitmap = originalBitmap;
      } else {
        delete (globalThis as Record<string, unknown>).createImageBitmap;
      }
    }
  });

  it('extracts a sequence from video frames and closes the detector', async () => {
    const originalDocument = (globalThis as Record<string, unknown>).document;
    const originalUrl = globalThis.URL;
    const revoked: string[] = [];
    (globalThis as Record<string, unknown>).document = {
      createElement() {
        const video: Record<string, unknown> = {
          duration: 0.75,
          muted: false,
          playsInline: false,
          onloadedmetadata: undefined,
          onerror: undefined,
          onseeked: undefined,
        };
        Object.defineProperty(video, 'src', {
          set() {
            queueMicrotask(() => (video.onloadedmetadata as (() => void) | undefined)?.());
          },
        });
        Object.defineProperty(video, 'currentTime', {
          set() {
            queueMicrotask(() => (video.onseeked as (() => void) | undefined)?.());
          },
        });
        return video;
      },
    };
    (globalThis as Record<string, unknown>).URL = {
      createObjectURL: () => 'blob:test-video',
      revokeObjectURL: (url: string) => revoked.push(url),
    } as unknown as URL;
    let calls = 0;
    let closed = false;
    const detector = {
      detect: () => ({}),
      detectForVideo: () => {
        calls += 1;
        return { landmarks: [landmarks(calls)] };
      },
      close: () => {
        closed = true;
      },
    } as VisionDetector;
    try {
      const sample = await extractMediaSample(
        new File(['video'], 'TWO-sequence.mp4', { type: 'video/mp4' }),
        undefined,
        async () => detector,
      );
      assert.equal(sample?.label, 'TWO');
      assert.equal(sample?.sequence?.length, 3);
      assert.equal(sample?.landmarks, sample?.sequence?.at(-1));
      assert.equal(calls, 3);
      assert.equal(closed, true);
      assert.deepEqual(revoked, ['blob:test-video']);
    } finally {
      if (originalDocument) {
        (globalThis as Record<string, unknown>).document = originalDocument;
      } else {
        delete (globalThis as Record<string, unknown>).document;
      }
      (globalThis as Record<string, unknown>).URL = originalUrl;
    }
  });
});