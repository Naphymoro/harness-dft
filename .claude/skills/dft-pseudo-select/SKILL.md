---
name: dft-pseudo-select
description: Install, list, and validate SSSP pseudopotential families for a given structure, functional, and protocol. Use before any DFT calculation on a structure/element combination that hasn't been checked yet, or when a run fails due to a missing pseudopotential.
---

# Pseudopotential selection and validation

## Checking what's installed

```bash
aiida-pseudo list
```

## Installing a family

```bash
aiida-pseudo install sssp --version 1.3 --functional PBE --protocol efficiency
# or --protocol precision for higher accuracy at higher cost
```

This can take a few minutes (downloads + parses ~100+ UPF files). Note the resulting label follows the pattern `SSSP/<version>/<functional>/<protocol>`, e.g. `SSSP/1.3/PBE/efficiency`.

## Validating coverage before a run

```python
from harness_dft.pseudos import validate_family_covers_structure, MissingPseudopotentialError

try:
    validate_family_covers_structure(atoms, "SSSP/1.3/PBE/efficiency")
except MissingPseudopotentialError as e:
    print(e)  # names exactly which elements are missing
```

All the workflow builders (`dft-relax`, `dft-converge`, `dft-bands-dos`) already call this internally and will raise `MissingPseudopotentialError` early with a clear message rather than letting a job fail deep inside a submitted QE calculation — if you see this error, install the missing coverage rather than trying to work around it.

## Known gotchas

- **Functional consistency**: `get_builder_from_protocol()`'s default protocols (`fast`/`balanced`/`stringent`) hardcode `SSSP/1.3/PBEsol/*` pseudo families by default — if you've only installed a `PBE` family (as this harness's local dev setup did), you **must** pass `overrides={'pseudo_family': 'SSSP/1.3/PBE/efficiency'}` (and for multi-stage workchains, to *every* sub-namespace: e.g. `{'base': {...}, 'base_final_scf': {...}}` for `PwRelaxWorkChain`, `{'scf': {...}, 'bands': {...}}` for `PwBandsWorkChain`). Missing one sub-namespace override is a real failure mode hit during development — the error message names exactly which family is missing, so read it rather than assuming the override took everywhere.
- Functional choice should match across structure, pseudopotentials, and any literature comparison — mixing PBE structure relaxation with PBEsol pseudopotentials for production numbers is a correctness footgun, not just a style choice.
