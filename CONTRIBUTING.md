# Contributing to AGI Pipeline

First off, thank you for considering contributing to the AGI Pipeline! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- **Check for existing issues**: Before opening a new issue, please search the issue tracker to see if the bug has already been reported.
- **Provide detail**: Include steps to reproduce the bug, expected behavior, and any relevant logs or screenshots.

### Suggesting Enhancements

- **Open an issue**: Describe the enhancement you'd like to see and why it would be useful.
- **Discuss**: Participate in the discussion to refine the idea.

### Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Install dependencies**: `pip install -r requirements.txt`.
3. **Follow the coding style**:
   - We use **Pylint** and **Flake8** for linting.
   - All modules, classes, and methods must have docstrings.
   - Follow PEP 8 guidelines.
   - Your code must achieve a 10.0/10 Pylint score.
4. **Write tests**: Ensure your changes are covered by tests in the `tests/` directory or as separate `test_*.py` files.
5. **Run tests**: Execute `pytest` to verify your changes.
6. **Submit the PR**: Provide a clear and descriptive pull request message.

## Development Environment

### Prerequisites

- Python 3.10+
- FFmpeg
- espeak-ng

### Testing

Run the test suite using:
```bash
pytest
```

### Linting

Check your code quality with:
```bash
pylint main.py
flake8 main.py
```

## License

By contributing, you agree that your contributions will be licensed under its [MIT License](LICENSE).
