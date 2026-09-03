# Changelog

All notable changes to NOESIS-Σ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial OSC dynamics engine implementation
- ICNN energy landscape with direct-gradient computation
- 24-layer foundation-scale native decoder
- Verification layer with NLI-based consistency checking
- Deterministic snapshotting and reproducibility features
- Basic validation test suite
- Core component architecture

### Changed
- Removed External dependencies for Core-only focus
- Simplified Runtime structure for experimental validation
- Updated documentation structure

### Removed
- API layer and application components
- External SIM/WKS/WorldModel integration
- Database dependencies (Redis, PostgreSQL, Qdrant)
- Frontend components
- Application deployment configurations

## [0.1.0-alpha.1] - 2026-09-03

### Added
- Initial experimental release
- Core OSC architecture
- Basic ICNN implementation
- Native decoder foundation
- Verification layer
- Documentation and guides

### Known Issues
- Limited empirical validation
- Experimental training procedures
- No large-scale benchmarking
- Performance characteristics unknown

### Experimental Status
- This is an alpha experimental release
- Not production-ready
- APIs and architectures may change
- Limited testing coverage
- Research-grade implementation

---

## Version Format

For this experimental research project, we use a simplified versioning:

- **Major**: Significant architectural changes
- **Minor**: New features and improvements
- **Patch**: Bug fixes and small improvements
- **alpha**: Experimental releases
- **beta**: More stable but still experimental
- **rc**: Release candidates
- **stable**: Production-ready releases

---

## Change Categories

### Added
- New features
- New components
- New documentation

### Changed
- Changes to existing functionality
- Refactoring
- Performance improvements

### Deprecated
- Soon-to-be removed features
- Alternatives provided

### Removed
- Removed features
- Removed dependencies

### Fixed
- Bug fixes
- Security fixes

### Security
- Security vulnerability fixes

---

## Release Process

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Create git tag
4. Push to GitHub
5. Create GitHub release

---

**Note**: This is experimental research software. Versioning may not follow strict semantic versioning during alpha phase.