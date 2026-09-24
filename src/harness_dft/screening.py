"""Cross-prototype comparison for "which candidate structure is the ground
state for this element" screening questions. Deliberately not a full
high-throughput campaign orchestrator (no job queue, no persistence beyond
what AiiDA already provides) -- just the one piece of analysis logic that
doesn't exist anywhere else: given several finished relaxations of the same
element in different candidate structures, rank them by energy per atom.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrototypeResult:
    label: str
    pk: int
    energy_ev: float
    n_atoms: int
    energy_per_atom_ev: float
    delta_from_lowest_ev_per_atom: float


@dataclass(frozen=True)
class RankedPrototypes:
    ranked: list[PrototypeResult] = field(default_factory=list)
    ground_state_label: str | None = None


def rank_prototypes(candidates: list[dict]) -> RankedPrototypes:
    """`candidates`: [{"label": "buckled-honeycomb", "pk": 573}, ...] -- each
    pk a FINISHED, successful relax/SCF job (PwRelaxWorkChain or
    PwBaseWorkChain) for the same element. Returns them sorted by energy per
    atom, lowest (most stable, at this level of theory) first.

    Only compares energies -- it does not itself check dynamical stability
    (see `harness_dft.stability`) or pseudopotential/cutoff consistency
    across candidates. Comparing energies computed with different ecutwfc,
    k-point density, or pseudopotential family is a correctness footgun the
    same way mixing functionals is (see the `dft-pseudo-select` skill) --
    this function trusts the caller to have held those fixed across
    `candidates`.
    """
    from harness_dft.jobs import get_results

    raw = []
    for candidate in candidates:
        results = get_results(candidate["pk"])
        params = results.get("output_parameters", {})
        energy_ev = params["energy"]
        n_atoms = params["number_of_atoms"]
        raw.append((candidate["label"], candidate["pk"], energy_ev, n_atoms, energy_ev / n_atoms))

    raw.sort(key=lambda row: row[4])
    lowest = raw[0][4] if raw else 0.0
    ranked = [
        PrototypeResult(
            label=label, pk=pk, energy_ev=energy_ev, n_atoms=n_atoms,
            energy_per_atom_ev=energy_per_atom, delta_from_lowest_ev_per_atom=energy_per_atom - lowest,
        )
        for label, pk, energy_ev, n_atoms, energy_per_atom in raw
    ]
    return RankedPrototypes(ranked=ranked, ground_state_label=ranked[0].label if ranked else None)
