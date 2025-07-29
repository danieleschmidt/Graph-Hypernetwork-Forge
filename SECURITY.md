# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please report security issues privately to:

**Email**: hypernetwork-forge-security@yourdomain.com

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 1 week
- **Fix Timeline**: Depends on severity and complexity

### Security Best Practices

When using this library:

1. **Keep dependencies updated**: Regularly update PyTorch and other dependencies
2. **Validate inputs**: Always validate data before processing
3. **Use secure environments**: Run training/inference in isolated environments
4. **Monitor resource usage**: Be aware of potential DoS through large graphs
5. **Secure model files**: Protect trained models from unauthorized access

### Vulnerability Categories

We are particularly interested in:

- Input validation bypasses
- Memory corruption issues
- Dependency vulnerabilities
- Denial of service vectors
- Model extraction attacks
- Training data poisoning

## Acknowledgments

We appreciate responsible disclosure and will acknowledge contributors in our security advisories (with permission).