# ENHANCEMENT-002: External Secrets Management Integration

**Status**: Proposed
**Priority**: P2 - Medium (Security)
**Effort**: Medium (12-20 hours)
**Type**: Security / Enterprise Feature
**Target Version**: 2.7.0 (Minor Release)

## Executive Summary

Add support for external secrets management systems (environment variables, HashiCorp Vault, AWS Secrets Manager, Azure Key Vault) to enable secure API key management in production environments. Currently, AbstractCore stores API keys in `~/.abstractcore/config/abstractcore.json`, which is suitable for development but not enterprise deployments.

**Expected Benefits**:
- Enterprise-grade secret management
- Compliance with security policies (SOC2, ISO 27001)
- Secret rotation support
- Audit trails for secret access
- Multi-environment configuration

---

## Problem Statement

### Current Limitations

**Storage**: API keys in plaintext JSON file
```json
// ~/.abstractcore/config/abstractcore.json
{
  "api_keys": {
    "openai": "sk-abc123...",
    "anthropic": "sk-ant-xyz..."
  }
}
```

**Issues**:
1. **Security Risk**: Plaintext storage
2. **No Rotation**: Manual key updates
3. **No Audit**: Can't track access
4. **Single User**: Not suitable for multi-user systems
5. **No Compliance**: Fails enterprise security audits

---

## Proposed Solution

### Multi-Tier Secrets Resolution

**Priority Order** (first match wins):
1. Explicit parameter: `api_key="sk-..."`
2. Environment variable: `OPENAI_API_KEY`
3. External secrets manager: Vault, AWS, Azure
4. Local config file: `~/.abstractcore/config/abstractcore.json`

### Architecture

```python
# abstractcore/config/secrets.py (NEW)

from abc import ABC, abstractmethod
from typing import Optional
import os

class SecretProvider(ABC):
    """Abstract base for secret providers."""

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """
        Retrieve secret value.

        Args:
            key: Secret key (e.g., "openai_api_key")

        Returns:
            Secret value or None if not found
        """
        pass

    @abstractmethod
    def supports(self, provider: str) -> bool:
        """Check if provider is supported."""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Get secrets from environment variables."""

    _KEY_MAPPING = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "huggingface": "HUGGINGFACE_TOKEN",
    }

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment."""
        env_var = self._KEY_MAPPING.get(key)
        if env_var:
            return os.getenv(env_var)
        return None

    def supports(self, provider: str) -> bool:
        """Check if provider has env var mapping."""
        return provider in self._KEY_MAPPING


class VaultSecretProvider(SecretProvider):
    """Get secrets from HashiCorp Vault."""

    def __init__(self, vault_addr: str, vault_token: str, path: str = "secret/data/abstractcore"):
        try:
            import hvac
            self.client = hvac.Client(url=vault_addr, token=vault_token)
            self.path = path
        except ImportError:
            raise ImportError("Install hvac for Vault support: pip install hvac")

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=self.path
            )
            return response['data']['data'].get(key)
        except Exception:
            return None

    def supports(self, provider: str) -> bool:
        """Vault supports all providers."""
        return True


class AWSSecretsProvider(SecretProvider):
    """Get secrets from AWS Secrets Manager."""

    def __init__(self, region: str = "us-east-1"):
        try:
            import boto3
            self.client = boto3.client('secretsmanager', region_name=region)
        except ImportError:
            raise ImportError("Install boto3 for AWS support: pip install boto3")

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(
                SecretId=f"abstractcore/{key}"
            )
            return response['SecretString']
        except Exception:
            return None

    def supports(self, provider: str) -> bool:
        """AWS supports all providers."""
        return True


class AzureKeyVaultProvider(SecretProvider):
    """Get secrets from Azure Key Vault."""

    def __init__(self, vault_url: str):
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            credential = DefaultAzureCredential()
            self.client = SecretClient(vault_url=vault_url, credential=credential)
        except ImportError:
            raise ImportError("Install Azure SDK: pip install azure-keyvault-secrets azure-identity")

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        try:
            secret = self.client.get_secret(key.replace("_", "-"))
            return secret.value
        except Exception:
            return None

    def supports(self, provider: str) -> bool:
        """Azure supports all providers."""
        return True


class SecretsManager:
    """
    Unified secrets management with multiple providers.

    Checks providers in order until secret is found.
    """

    def __init__(self):
        self.providers: List[SecretProvider] = []

        # Always include environment provider
        self.add_provider(EnvironmentSecretProvider())

    def add_provider(self, provider: SecretProvider):
        """Add a secret provider."""
        self.providers.append(provider)

    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for provider.

        Checks all registered providers in order.

        Args:
            provider_name: Provider name (e.g., "openai")

        Returns:
            API key or None
        """
        for provider in self.providers:
            if provider.supports(provider_name):
                key = provider.get_secret(provider_name)
                if key:
                    return key

        # Fallback to config file
        from .manager import get_config_manager
        config_manager = get_config_manager()
        return config_manager.get_api_key(provider_name)


# Global secrets manager
_secrets_manager = SecretsManager()

def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager."""
    return _secrets_manager
```

### Integration with Providers

```python
# abstractcore/providers/openai_provider.py

from ..config.secrets import get_secrets_manager

class OpenAIProvider(BaseProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        # Priority order:
        # 1. Explicit parameter
        # 2. Secrets manager (env, vault, aws, azure)
        # 3. Config file
        if api_key is None:
            api_key = get_secrets_manager().get_api_key("openai")

        if not api_key:
            raise AuthenticationError("openai", "No API key found")

        self.api_key = api_key
        # ... rest of initialization
```

### Configuration via CLI

```bash
# Environment variables (simplest)
export OPENAI_API_KEY=sk-abc123
export ANTHROPIC_API_KEY=sk-ant-xyz

# Configure Vault
abstractcore --set-secrets-provider vault \
    --vault-addr https://vault.company.com \
    --vault-token hvs.token \
    --vault-path secret/data/abstractcore

# Configure AWS Secrets Manager
abstractcore --set-secrets-provider aws \
    --aws-region us-east-1

# Configure Azure Key Vault
abstractcore --set-secrets-provider azure \
    --vault-url https://myvault.vault.azure.net
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (4-6 hours)

1. Create `abstractcore/config/secrets.py`
2. Implement `SecretProvider` ABC
3. Implement `EnvironmentSecretProvider`
4. Implement `SecretsManager`
5. Write tests

### Phase 2: External Providers (4-6 hours)

1. Implement `VaultSecretProvider` (with hvac)
2. Implement `AWSSecretsProvider` (with boto3)
3. Implement `AzureKeyVaultProvider` (with azure-sdk)
4. Make all optional dependencies
5. Write integration tests

### Phase 3: Provider Integration (2-3 hours)

1. Update all 6 providers to use `SecretsManager`
2. Maintain backwards compatibility
3. Test secret resolution order
4. Verify fallback to config file

### Phase 4: CLI & Documentation (2-3 hours)

1. Add CLI commands for secrets configuration
2. Update `abstractcore --status` to show secrets source
3. Document usage patterns
4. Create examples for each provider

### Phase 5: Testing (2-4 hours)

1. Unit tests for each provider
2. Integration tests with mocked services
3. Security tests (no leaks in logs)
4. Fallback chain tests

**Total Estimated Time**: 14-22 hours

---

## Security Considerations

### Secret Handling Best Practices

1. **Never Log Secrets**:
```python
logger.info("Using API key", key_source="environment")  # ✅ Log source
logger.info(f"API key: {api_key}")  # ❌ Never log value
```

2. **Mask in Errors**:
```python
def mask_secret(secret: str) -> str:
    """Mask secret for display."""
    if len(secret) < 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"

# Use in errors
raise AuthenticationError(f"Invalid key: {mask_secret(api_key)}")
```

3. **Secure Storage**:
```python
# Config file should have restrictive permissions
import os
os.chmod(config_file, 0o600)  # Owner read/write only
```

4. **Audit Logging**:
```python
logger.info("Secret accessed",
    provider="openai",
    source="vault",
    user=os.getenv("USER"),
    timestamp=datetime.now().isoformat()
)
```

---

## Success Criteria

1. **Multi-Source Support**: Environment, Vault, AWS, Azure
2. **Priority Resolution**: Correct fallback chain
3. **Zero Leaks**: No secrets in logs or errors
4. **Backwards Compatible**: Config file still works
5. **Documentation**: Complete setup guides
6. **Security**: Passes security audit

---

## Testing

```python
# tests/config/test_secrets_management.py

def test_environment_provider():
    """Test environment variable resolution."""
    import os
    os.environ["OPENAI_API_KEY"] = "test-key-123"

    provider = EnvironmentSecretProvider()
    key = provider.get_secret("openai")

    assert key == "test-key-123"

def test_priority_order():
    """Verify resolution priority."""
    # 1. Explicit param (highest)
    llm = create_llm("openai", model="gpt-4", api_key="explicit-key")
    assert llm.api_key == "explicit-key"

    # 2. Environment (if no explicit)
    os.environ["OPENAI_API_KEY"] = "env-key"
    llm = create_llm("openai", model="gpt-4")
    assert llm.api_key == "env-key"

    # 3. Config file (fallback)
    del os.environ["OPENAI_API_KEY"]
    # ... (requires config file setup)

def test_vault_integration():
    """Test Vault integration (mock)."""
    # Mock Vault client
    with mock.patch('hvac.Client') as mock_vault:
        mock_vault.return_value.secrets.kv.v2.read_secret_version.return_value = {
            'data': {'data': {'openai': 'vault-key-123'}}
        }

        provider = VaultSecretProvider("http://vault", "token")
        key = provider.get_secret("openai")

        assert key == "vault-key-123"

def test_no_secret_leaks_in_logs():
    """Verify secrets never appear in logs."""
    with capture_logs() as logs:
        llm = create_llm("openai", model="gpt-4", api_key="sk-secret123")

    # Secret should not appear in logs
    assert "sk-secret123" not in str(logs)
    # But source should be mentioned
    assert "explicit" in str(logs) or "parameter" in str(logs)
```

---

## Documentation

### Environment Variables Guide

```markdown
## Using Environment Variables (Recommended)

The simplest approach for production:

```bash
# Set API keys as environment variables
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key

# AbstractCore will automatically use them
python your_script.py
```

Supports:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `HUGGINGFACE_TOKEN`
- `LMSTUDIO_API_KEY`
```

### HashiCorp Vault Guide

```markdown
## Using HashiCorp Vault

For enterprise deployments:

```bash
# Configure Vault
abstractcore --set-secrets-provider vault \
    --vault-addr https://vault.company.com \
    --vault-token $VAULT_TOKEN \
    --vault-path secret/data/abstractcore

# Store secrets in Vault
vault kv put secret/abstractcore \
    openai=sk-your-key \
    anthropic=sk-ant-your-key

# AbstractCore will retrieve from Vault
python your_script.py
```
```

---

## Optional Dependencies

```toml
# pyproject.toml

[project.optional-dependencies]
secrets-vault = [
    "hvac>=1.0.0,<2.0.0",  # HashiCorp Vault
]

secrets-aws = [
    "boto3>=1.26.0,<2.0.0",  # AWS Secrets Manager
]

secrets-azure = [
    "azure-keyvault-secrets>=4.6.0,<5.0.0",
    "azure-identity>=1.12.0,<2.0.0",
]

secrets-all = [
    "abstractcore[secrets-vault,secrets-aws,secrets-azure]",
]
```

---

## Rollout Plan

1. **Phase 1**: Environment variables (immediately useful)
2. **Phase 2**: Vault (most requested enterprise feature)
3. **Phase 3**: AWS & Azure (cloud platforms)
4. **Phase 4**: Documentation and examples

---

## References

- HashiCorp Vault: https://www.vaultproject.io/
- AWS Secrets Manager: https://aws.amazon.com/secrets-manager/
- Azure Key Vault: https://azure.microsoft.com/en-us/services/key-vault/
- 12-Factor App: https://12factor.net/config

---

**Document Version**: 1.0
**Created**: 2025-11-25
**Author**: Expert Code Review
**Status**: Ready for Review
