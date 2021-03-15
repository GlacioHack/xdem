## How to contribute
Contributors of `xdem` should attempt to conform to pep8 coding standards.
An exception to the standard is having a 120 max character line length (instead of 80).

Suggested linters are:
1. prospector
2. mypy (git version)
3. pydocstyle

Suggested formatters are:
1. autopep8
2. isort

These can all be installed with this command:
```bash
pip install prospector git+https://github.com/mypy/mypy.git pydocstyle autopep8 isort
```
Note that your text editor of choice will also need to be configured with these tools (and max character length changed).

Contributions are welcome in the form of pull requests (PRs) to the main branch.
At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, but more than one is suggested.
