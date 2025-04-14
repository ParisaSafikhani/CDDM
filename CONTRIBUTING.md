# Contributing to CDDM

Thank you for considering contributing to CDDM! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. **Fork the repository**: Create your own copy of the repository to work on.

2. **Create a branch**: Create a branch for your changes.
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**: Implement your changes, following the code style guidelines below.

4. **Test your changes**: Ensure your changes don't break existing functionality.

5. **Commit your changes**: Write clear, concise commit messages.
   ```bash
   git commit -m "Add feature X"
   ```

6. **Push to your fork**: Upload your changes to your fork.
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request**: Submit a pull request from your fork to the main repository.

## Code Style Guidelines

- Follow PEP 8 style guidelines for Python code.
- Use meaningful variable and function names.
- Include docstrings for all functions, classes, and modules.
- Write clear comments for complex sections of code.

## Adding New Features

When adding new features, please consider the following:

1. **Domain Additions**: If adding new domains to the classifier, ensure they are well-defined and come with clear descriptions.

2. **Model Selector Enhancements**: When enhancing the model selection algorithm, provide benchmarks to demonstrate improvements.

3. **AutoML Integration**: For changes to the AutoPyTorch integration, ensure compatibility with the latest AutoPyTorch version.

## Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- System information (Python version, OS, etc.)

## Feature Requests

Feature requests are welcome! Please provide:

- A clear description of the feature
- Motivation for the feature
- Potential implementation approaches (if applicable)

## Questions

If you have questions about the project, please open an issue with the "question" label.

Thank you for contributing to CDDM! 