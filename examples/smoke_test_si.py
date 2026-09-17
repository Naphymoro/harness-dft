"""Minimal end-to-end validation: relax bulk Si and print the equilibrium
lattice parameter and energy. Run after setting up the environment (see
README.md) with an AiiDA profile loaded, a `pw.x` code registered, and the
SSSP pseudopotential family installed.

    python examples/smoke_test_si.py
"""
from aiida import load_profile
from ase.build import bulk

from harness_dft.workflows.relax import run_relax

load_profile()

atoms = bulk("Si", "diamond", a=5.45)
results, node, plan = run_relax(
    atoms,
    code_label="pw-7.5@localhost",
    protocol="fast",
    kpoints_mesh=(4, 4, 4),
    ecutwfc_ry=40.0,
)

print(f"Execution plan: {plan}")
print(f"Exit status: {node.exit_status}")
if node.is_finished_ok:
    energy_ev = results["output_parameters"].get_dict()["energy"]
    relaxed = results["output_structure"].get_ase()
    print(f"Final energy: {energy_ev:.4f} eV")
    print(f"Relaxed cell parameters: {relaxed.cell.cellpar()}")
else:
    print(f"Relaxation failed -- run `verdi process report {node.pk}` for details.")
