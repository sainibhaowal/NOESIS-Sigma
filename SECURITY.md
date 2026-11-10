# Security Policy

## Scope

NOESIS-Σ / SigmaStorm is research software executed locally or in user-controlled environments
such as Kaggle. The policy covers the source code, scripts, configuration, documentation, and
release artifacts in this repository.

The repository does not provide a hosted service or process credentials on behalf of users. Reports
involving third-party services, datasets, package registries, or Kaggle infrastructure should be
reported to the relevant provider as well as to this project when the repository code is involved.

## Supported versions

| Version | Support |
| --- | --- |
| `main` | Best-effort security fixes |
| `0.12.0` | Best-effort security fixes |
| Older versions | Not routinely maintained |

## Reporting a vulnerability

Please do not disclose credentials, exploit details, or other sensitive information in a public
issue. Use GitHub private vulnerability reporting or a private Security Advisory for the canonical
repository when that feature is available. If private reporting is unavailable, contact the
maintainer through the [Ravinder Singh GitHub profile](https://github.com/sainibhaowal) and include
only the minimum information needed to establish private contact.

Please include:

- the affected version or commit;
- the affected file, command, or configuration;
- a minimal reproduction and expected impact;
- whether the issue requires a local checkpoint, dataset, network access, or credentials; and
- a suggested mitigation, if known.

The maintainer will assess reports on a best-effort basis. Public disclosure should wait until a
fix or mitigation is available, or until the reporter and maintainer agree that disclosure is safe.

## Security guidance for users

- Never commit API keys, Kaggle credentials, `.env` files, private datasets, or private checkpoints.
- Treat model checkpoints as untrusted input. `scripts/eval.py` currently loads checkpoints with
  PyTorch's general deserialization path (`weights_only=False`); load only checkpoints you trust.
- Review external scripts, datasets, and package versions before enabling network access.
- The ignored `data/` and `kaggle/` directories are local-workspace boundaries, not a substitute for
  rotating a credential that has already been exposed.

For contribution-related questions, see [CONTRIBUTING.md](CONTRIBUTING.md). For ordinary usage
questions, see [SUPPORT.md](SUPPORT.md).
