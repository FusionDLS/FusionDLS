# FusionDLS Development

## Installation from source

To install from source, it is recommended to set up a dedicated Python environment.
This can be achieved using `venv`:

```bash
# In the top level of the project...
python -m venv venv  # Create a virtual environment called 'venv'
source venv/bin/activate  # Activate it
```

After finishing your work on FusionDLS, you can leave this envionment using:

```bash
deactivate
```

To install, we recommend using the following:

```bash
# In the top level of the project...
pip install -e .[tests,docs,linting]
```

- The `-e` installs in editable mode, so any changes to the code are immediately
  reflected in your Python environment.
- The `.` installs the project in your current working directory.
- The terms in square brackets list groups of optional dependencies defined in
  the `pyproject.toml` file.

## Testing

Unit tests can be run by calling:

```bash
# In the top level of the project...
pytest
```

There are numerous options to control the output of `pytest`:


## Linting and Formatting

When contributing, please use Ruff for linting and formatting. This is available as a
VSCode plugin, but it can be easily run from the command line too:

```bash
ruff format  # Ensures consistent code format throughout the project
ruff check   # Checks for style issues, possible bugs, etc
```

Some warnings raised by Ruff can be automatically fixed:

```bash
ruff check --fix
```

To avoid accidentally pushing unlinted/unformatted code to GitHub, we recommend using
the Git pre-commit hook provided:

```bash
git config --local core.hooksPath .githooks
```

This will run the formatter, check for linter warnings and run the tests before allowing
you to commit. To override the pre-commit hook:

```bash
git commit --no-verify
# Or...
git commit -n
```

## Building the Documentation

The online documentation can be built locally using:

```bash
# From the top level of the project...
cd docs
make html
```

To inspect the docs, open the file
`/path/to/FusionDLS/docs/_build/html/index.html` in your web browser.

Sometimes the docs may fail to build, especially after making substantive
changes to the code such as renaming/removing files. Often this may be fixed by
leaning the local build and deleting any automatically generated files:

```bash
cd docs
make clean
rm generated/*
```