# ISL Neuro-Fuzzy Recognizer

An interactive research workbench for the hybrid ANFIS + GRU Indian Sign Language digit recognizer.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/isl-recognizer/src/pages/` — workbench, training, and evaluation views
- `artifacts/isl-recognizer/src/components/` — shared shell and landmark visualization
- `artifacts/api-server/src/lib/isl-model.ts` — feature extraction, fuzzy inference, temporal smoothing, and synthetic training
- `artifacts/api-server/src/routes/` — recognition, training, evaluation, and health routes
- `lib/api-spec/openapi.yaml` — source of truth for the API contract

## Architecture decisions

- The first release is dataset-free by design: the API exposes a deterministic synthetic landmark sequence so the complete pipeline can be explored immediately.
- Spatial inference uses the deck's three interpretable descriptors: finger count, normalized hand spread, and wrist-axis orientation.
- The temporal stage uses reliability-gated recurrent smoothing over per-frame ANFIS posteriors; it is intentionally CPU-friendly for the workbench.
- Training is exposed as a reproducible simulation until a labeled dataset is supplied, with the deck's 14-rule ANFIS and 16-unit GRU configuration represented in the result.

## Product

The app lets users run a synthetic 12-frame ISL digit sequence through the recognizer, inspect its 21-point landmark field and posterior distribution, run dual-stage training experiments with optional geometry augmentation, and compare baseline versus proposed evaluation metrics.

## User preferences

No additional project-specific preferences recorded.

## Gotchas

- Regenerate API hooks and Zod schemas after changing `lib/api-spec/openapi.yaml`.
- The evaluation values are deck-aligned reference metrics, not claims of validation on an uploaded production dataset.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
