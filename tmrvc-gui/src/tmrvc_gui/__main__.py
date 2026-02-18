"""Entry point for `uv run tmrvc-gui` or `python -m tmrvc_gui`."""

from tmrvc_gui.app import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
