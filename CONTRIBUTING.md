# Contributing to NOESIS-Σ

Thank you for your interest in contributing to NOESIS-Σ! This is an experimental research project, and we welcome contributions that advance the science and engineering of differential equation-based cognition.

## 🎯 Contribution Areas

We're particularly interested in contributions to:

### Theoretical Research
- Mathematical analysis of OSC convergence properties
- Theoretical guarantees for ICNN energy landscapes
- Stability analysis of operator-split methods
- Novel differential equation approaches to cognition

### Empirical Validation
- New reasoning benchmarks and evaluation metrics
- Ablation studies on architectural components
- Comparative studies with transformer baselines
- Scaling studies and performance analysis

### Implementation
- Training algorithm improvements
- Performance optimization and efficiency gains
- Numerical stability enhancements
- Reproducibility and testing improvements

### Documentation
- Tutorial and guide improvements
- API documentation enhancements
- Research paper writing and editing
- Educational content creation

## 🚀 Getting Started

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/sainibhaowal/NOESIS-Sigma.git
cd NOESIS-Sigma

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r Requirements/requirements.txt
pip install pytest black mypy isort
```

### 2. Understand the Architecture

Read the documentation in `Docs/` to understand:
- OSC dynamics engine architecture
- ICNN energy landscape design
- Core component interactions
- Training and validation procedures

### 3. Run Existing Tests

```bash
# Run unit tests
pytest Tests/core/

# Run integration tests
pytest Tests/integration/

# Run validation tests
python Runtime/Validation/core_test.py
```

## 📝 Contribution Guidelines

### Code Standards

- **Python**: Follow PEP 8 style guidelines
- **Type Hints**: Add type hints to all functions
- **Documentation**: Include docstrings for all public functions
- **Testing**: Add unit tests for new functionality
- **Formatting**: Use Black for code formatting

### Commit Guidelines

- Write clear, descriptive commit messages
- Reference related issues with `#issue-number`
- Keep commits focused and atomic
- Use conventional commit format: `type: description`

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes following code standards
4. **Add** tests for new functionality
5. **Update** documentation as needed
6. **Push** to your fork (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request with clear description

### PR Requirements

- **Clear description** of what you're changing and why
- **Related issues** referenced
- **Tests added** for new functionality
- **Documentation updated** for API changes
- **All tests passing** in CI/CD

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest Tests/

# Run specific test file
pytest Tests/core/test_dynamics.py

# Run with coverage
pytest Tests/ --cov=Core --cov-report=html
```

### Writing Tests

- Use pytest for all tests
- Mock external dependencies
- Test both success and failure cases
- Keep tests independent and fast

## 📚 Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Explain the "why" not just the "how"
- Keep documentation up to date with code changes

### Documentation Areas

- **API docs**: Function/class descriptions
- **Tutorials**: Step-by-step guides
- **Research papers**: Academic publications
- **Architecture docs**: System design documentation

## 🤝 Research Contributions

### Academic Standards

- **Cite related work** appropriately
- **Provide reproducible experiments**
- **Share evaluation metrics** and methodologies
- **Document experimental setup** clearly

### Publication Process

- Discuss large research contributions via issues first
- Follow academic standards for research papers
- Include proper attribution and citations
- Share data and code for reproducibility

## 🐛 Bug Reports

### Reporting Bugs

1. **Search** existing issues first
2. **Use** the bug report template
3. **Provide** minimal reproducible example
4. **Include** environment details (Python version, dependencies)
5. **Describe** expected vs actual behavior

### Bug Report Template

```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- Python version:
- OS:
- Dependencies:

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Proposing Features

1. **Check** existing issues and PRs
2. **Explain** the use case and motivation
3. **Describe** the proposed solution
4. **Consider** alternatives and trade-offs
5. **Be** open to discussion and iteration

### Feature Request Template

```markdown
**Problem Description**
What problem does this solve?

**Proposed Solution**
How should this be implemented?

**Alternatives Considered**
What other approaches did you consider?

**Additional Context**
Any other relevant information
```

## 📜 Code of Conduct

### Our Pledge

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Respect differing opinions and experiences

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community

### Enforcement

Project maintainers reserve the right to remove content or ban users who violate these standards.

## 🔐 Security

### Security Issues

- **DO NOT** report security issues via public GitHub issues
- **DO** report security issues privately via [GitHub Security Advisories](https://github.com/sainibhaowal/NOESIS-Sigma/security/advisories)
- **DO** include detailed description and reproduction steps
- **DO** allow time for the issue to be addressed

See [SECURITY.md](SECURITY.md) for details.

## 📅 Project Status

This is an **alpha experimental** research project:

- **APIs may change** without notice
- **Features may be added/removed** rapidly
- **Documentation may be incomplete**
- **Testing coverage is limited**

Contributors should be aware of the experimental nature and potential for instability.

## 🎓 Academic Credit

### Authorship

For significant research contributions:
- Discuss authorship early in the process
- Follow academic standards for attribution
- Include contributors in related publications
- Provide proper citations and acknowledgments

### Publication

- Discuss publication plans for major contributions
- Follow academic publication standards
- Include all contributors appropriately
- Share preprints and maintain citations

## 📞 Getting Help

### Questions

- **GitHub Discussions**: For general questions and research collaboration
- **GitHub Issues**: For bugs and feature requests

### Resources

- [Core Engine](Core/)
- [Test Suite](Tests/)
- [Changelog](CHANGELOG.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## 🙏 Recognition

All contributors will be:
- Listed in the CONTRIBUTORS file
- Acknowledged in release notes
- Cited in related publications
- Invited to contribute to research papers

Thank you for contributing to the future of neural reasoning!

---

**Note**: This is experimental research software. By contributing, you acknowledge the experimental nature and potential instability of the project.