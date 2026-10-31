# Security

Handoff runs on your machines and handles footage that may show people and workplaces. We take that seriously.

- **No network access needed.** Handoff Check makes no network calls and sends no telemetry. The Docker image is
  meant to run with `--network none`.
- **No data leaves your environment.** Reports are written only where you point `--out`.
- **No biometric templates are stored.** Hand detection produces counts and confidence values per sampled frame,
  nothing that identifies a person.

## Reporting a vulnerability

Please report it privately with the "Report a vulnerability" button on this repository's Security tab, with details and steps to reproduce. Don't open a public issue. We'll acknowledge
within 3 business days.
