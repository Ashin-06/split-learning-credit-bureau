# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately, through GitHub's private vulnerability
reporting on this repository (Security → Report a vulnerability) or by contacting the
maintainer, [@Ashin-06](https://github.com/Ashin-06), directly. Do not open a public issue.

Include what you found, how to reproduce it and what an attacker could do with it. You should
get an acknowledgement within 5 working days. Please give us 90 days to fix an issue before
disclosing it, and don't test against anyone else's deployment.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no (prototype releases) |

## Scope

In scope: the bureau service, the private set intersection (`psi.py`), the split network and its
cut-layer defence, the model registry and score log, the container image and the CI configuration
in this repository.

Out of scope: your reverse proxy or TLS setup, and findings that need an already-compromised host
or a leaked API key with the matching role.

**The PSI protocol is unreviewed cryptography.** It uses libsodium's Ed25519 group arithmetic
(through PyNaCl), but the protocol code is hand-written, has not been reviewed by a cryptographer,
is secure only against parties that follow it (semi-honest), and hashes identifiers onto the curve
in non-constant time. The service logs a warning saying so when it starts in production mode, and
`/psi/align` returns the same warning. Do not use it to match real customer identifiers.

The cut-layer defence has no formal privacy guarantee (no differential-privacy accounting); see
`docs/threat-model.md`.

This project has **not** had an independent security review or penetration test.
