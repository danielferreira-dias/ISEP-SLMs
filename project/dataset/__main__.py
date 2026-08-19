"""Allow ``uv run python -m project.dataset`` to smoke-load the Hub dataset."""

from project.dataset.load import main

if __name__ == "__main__":
    main()
