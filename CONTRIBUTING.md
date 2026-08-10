# Contributing

Issues and pull requests are welcome, in English or Portuguese.

## Development setup

```bash
git clone https://github.com/cesarrlamaral/hydrovent_field.git
cd hydrovent_field
pip install -r requirements.txt
pytest tests/
```

## Guidelines

- **Physical claims need a citation.** This project's standard (see
  [`docs/PHYSICS_MODEL.md`](docs/PHYSICS_MODEL.md)) is that every equation,
  constant, or parameter range is traceable to a primary source, and every
  illustrative/uncalibrated choice is flagged explicitly rather than
  presented as validated. New physical modules or parameter changes should
  follow the same standard: cite the source, add a benchmark test against a
  real measured value when one exists, and document what remains
  uncalibrated or speculative.
- **Add a regression test for bug fixes**, and a benchmark test for new
  physics, in `tests/`.
- **Reproducibility**: any change to seed derivation, sampling order, or
  parameter defaults is a breaking change for anyone relying on exact
  reproducibility of a past run — call it out explicitly in the PR
  description.
- Run `pytest tests/` before submitting; CI runs the same suite on Python
  3.10–3.12.

## Reporting issues

Please include: the exact command/GUI action that produced the problem,
the full traceback (if any), and — for anything involving simulation output —
the `--seed` used, so the run is reproducible.
