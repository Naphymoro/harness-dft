"""Entry point: python -m harness_dft_mcp [--transport stdio|streamable-http].

Needs an AiiDA profile resolvable in this environment (set AIIDA_PROFILE, or
have exactly one default profile) and, for local execution, QE binaries
registered as AiiDA codes -- see the harness-dft README for setup, and the
`dft-harness` DeerFlow skill for how an agent should drive this server.
"""
import argparse
import logging
import sys

from .server import create_server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harness-dft-mcp", description="harness-dft MCP server")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http only; there is no authentication, keep it local")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args(argv)
    # stdout carries the MCP protocol on stdio; every log line must go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    create_server(host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
