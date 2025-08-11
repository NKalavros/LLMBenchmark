"""Module entrypoint for `python -m agent_benchmark`.

Delegates to cli.main().
"""
from .cli import main


if __name__ == "__main__":  # pragma: no cover
    main()
