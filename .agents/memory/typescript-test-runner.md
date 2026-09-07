---
name: TypeScript test runner
description: How to run tests that import this workspace's source modules directly.
---

Use the workspace's installed tsx runner for source-level Node tests. Node's native TypeScript stripping does not resolve the extensionless workspace imports used by the bundler configuration.

**Why:** Native Node test execution failed before any test ran because package and relative imports are intentionally written for the workspace bundler.

**How to apply:** Keep test scripts on the existing workspace runner rather than rewriting production imports only to satisfy Node's resolver.