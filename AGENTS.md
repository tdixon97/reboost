# OpenAI Codex

## Testing requirements

Install all required dependencies for tests, linting and docs building by using
the `all` optional dependency group defined in `pyproject.toml`. With
python-pip, you would need to run `pip install -e '.[all]'`.

Use Pytest to run all unit tests.

## Linting

To run linting, use pre-commit. Always make sure pre-commit checks pass before
committing.
