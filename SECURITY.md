# Security Policy

## Supported Versions

We provide security updates for the following versions of PrivaStream:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of PrivaStream seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please send an email to security@privastream.ai with the following information:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes (if available)

### What to Expect

- We will acknowledge receipt of your vulnerability report within 48 hours
- We will provide a detailed response within 7 days indicating next steps
- We will keep you informed of our progress towards fixing the vulnerability
- We may ask for additional information or guidance during our investigation

### Security Best Practices

When using PrivaStream:

1. **Data Privacy**: Ensure processed video/audio data is handled according to your privacy requirements
2. **Model Security**: Only use trusted model files from verified sources
3. **Network Security**: Secure WebRTC connections in production environments
4. **Access Control**: Implement proper authentication for web interfaces
5. **Updates**: Keep dependencies updated to patch known vulnerabilities

### Scope

This security policy applies to:
- Core PrivaStream application code
- Web interface components
- AI model inference pipelines
- Configuration and deployment scripts

### Out of Scope

- Third-party dependencies (report to upstream maintainers)
- Issues requiring physical access to the system
- Social engineering attacks
- Issues in development or testing environments

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any similar problems
3. Prepare fixes for supported versions
4. Release security patches as soon as possible
5. Publish a security advisory on GitHub

## Recognition

We appreciate the security research community's efforts to improve the security of open source projects. Contributors who responsibly disclose security vulnerabilities will be acknowledged in our security advisories (unless they prefer to remain anonymous).
