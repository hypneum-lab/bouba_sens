# Security Policy

## Supported Versions

As of Sprint 0, `bouba_sens` is pre-release (v0.0.x). Only the `main` branch
is security-supported.

## Reporting a Vulnerability

Please report vulnerabilities privately to `c.saillant@gmail.com` with the
subject line `[bouba_sens security] <summary>`. Do **not** open a public
issue for unpatched vulnerabilities.

We aim to acknowledge reports within 5 business days and to provide an
initial assessment within 15 business days.

## Scope

`bouba_sens` is a research benchmark. Relevant vulnerabilities include:

- Arbitrary code execution via malicious config files (YAML, pickle) ingested
  through `bouba-sens run` / `bouba-sens eval`.
- Privilege escalation through install-time hooks.
- Supply-chain compromise via a pinned dependency.

Out of scope: incorrect research claims, missing features, or performance
issues.
