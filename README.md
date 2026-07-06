# Calculable R-matrix and Numerov Benchmarks

A learning project reproducing selected calculable R-matrix examples from Descouvemont and Baye's review of R-matrix theory, with comparisons to direct Numerov integration where appropriate.

The goal is to build a transparent numerical understanding of how finite-basis R-matrix calculations reproduce scattering observables such as phase shifts.

## Current focus

This project currently focuses on the Section 4 examples from Descouvemont and Baye.

Planned examples:

- Narrow resonance: \(^{12}\mathrm{C}+p\), \(l=0\)
- Broad resonance: \(\alpha+\alpha\), \(l=4\)
- Non-resonant scattering: \(\alpha+{}^3\mathrm{He}\), \(l=0\)

## Repository structure

```text
notebooks/   exploratory notebooks and reproduced calculations
src/         reusable Python code
figures/     generated plots
docs/        progress notes and additional documentation
```

## Status

Current focus: implementing the \(^{12}\mathrm{C}+p\), \(l=0\) narrow-resonance example and benchmarking against Numerov integration. 

See `docs/progress.md` for occasional development notes.

- [x] Set up project structure
- [ ] Implemented basic potentials
- [ ] Implement Numerov solver
- [ ] Extract phase shifts
- [ ] Implement calculable R-matrix method