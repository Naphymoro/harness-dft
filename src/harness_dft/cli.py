"""Minimal CLI entry point. Requires an AiiDA profile to already be loaded
in the environment (set AIIDA_PROFILE or have exactly one default profile)."""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_resources(args):
    from harness_dft.environment import get_local_resources
    print(json.dumps(get_local_resources().__dict__, indent=2))


def _cmd_relax(args):
    from aiida import load_profile
    from ase.io import read

    load_profile()
    from harness_dft.workflows.relax import run_relax

    atoms = read(args.structure)
    results, node, plan = run_relax(
        atoms, code_label=args.code,
        pseudo_family_label=args.pseudo_family,
        kpoints_mesh=tuple(args.kpoints),
        ecutwfc_ry=args.ecutwfc,
    )
    print(f"Execution plan: {plan}")
    print(f"Exit status: {node.exit_status}")
    if node.is_finished_ok:
        print(f"Final energy (eV): {results['output_parameters'].get_dict()['energy']}")
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="harness-dft")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("resources", help="print detected local CPU/RAM/GPU").set_defaults(func=_cmd_resources)

    relax_parser = subparsers.add_parser("relax", help="relax a structure via PwRelaxWorkChain")
    relax_parser.add_argument("structure", help="path to a structure file readable by ase.io.read")
    relax_parser.add_argument("--code", required=True, help="AiiDA code label or pk for pw.x")
    relax_parser.add_argument("--pseudo-family", dest="pseudo_family", default="SSSP/1.3/PBE/efficiency")
    relax_parser.add_argument("--kpoints", nargs=3, type=int, default=[4, 4, 4])
    relax_parser.add_argument("--ecutwfc", type=float, default=40.0)
    relax_parser.set_defaults(func=_cmd_relax)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
