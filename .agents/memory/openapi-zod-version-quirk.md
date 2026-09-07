---
name: OpenAPI integer schema compatibility
description: Orval's generated Zod output currently targets Zod 3 in this workspace.
---

When adding integer fields to the shared OpenAPI contract, verify the generated Zod output against the installed Zod version; plain integer schemas may emit `zod.int()`, which is unavailable in this workspace's Zod 3 runtime.

**Why:** Code generation can succeed while the chained library typecheck fails on generated validators, blocking the rest of the build.

**How to apply:** Prefer a compatible numeric schema when integer validation is not essential, or confirm the workspace Zod/Orval versions before introducing integer fields.