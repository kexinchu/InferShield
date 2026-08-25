"""Optional bearer-token binding for SafeKV tenant principals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


class PrincipalAuthenticationError(ValueError):
    """Raised when a configured principal binding cannot authenticate a request."""


SamplingParams = Optional[Union[Dict, List[Dict]]]


def bind_openai_user_id(request: Any, effective_principal: Optional[str]) -> None:
    """Apply a trusted principal to a parsed OpenAI request when enabled."""
    if effective_principal is not None:
        request.user_id = effective_principal


@dataclass(frozen=True)
class PrincipalBinding:
    """Maps opaque bearer tokens to server-trusted principal identifiers."""

    token_to_principal: Optional[Mapping[str, str]] = None

    @property
    def enabled(self) -> bool:
        return self.token_to_principal is not None

    @classmethod
    def from_json_file(cls, path: Optional[str]) -> "PrincipalBinding":
        if path is None:
            return cls()

        with Path(path).open(encoding="utf-8") as file:
            mapping = json.load(file)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Principal binding file must contain a non-empty JSON object")
        if any(
            not isinstance(token, str)
            or not token
            or not isinstance(principal, str)
            or not principal
            for token, principal in mapping.items()
        ):
            raise ValueError(
                "Principal binding tokens and principal IDs must be non-empty strings"
            )
        return cls(token_to_principal=mapping)

    def authenticate(self, authorization: Optional[str]) -> Optional[str]:
        """Return the trusted principal, or None when binding is disabled."""
        if not self.enabled:
            return None
        if not authorization:
            raise PrincipalAuthenticationError("Bearer authentication required")

        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not token
            or token.strip() != token
        ):
            raise PrincipalAuthenticationError("Invalid bearer credentials")

        principal = self.token_to_principal.get(token)
        if principal is None:
            raise PrincipalAuthenticationError("Invalid bearer credentials")
        return principal

    def bind_sampling_params(
        self, sampling_params: SamplingParams, principal: str
    ) -> SamplingParams:
        """Overwrite client-controlled user IDs before scheduler submission."""
        if sampling_params is None:
            return {"user_id": principal}
        if isinstance(sampling_params, list):
            for params in sampling_params:
                params["user_id"] = principal
        else:
            sampling_params["user_id"] = principal
        return sampling_params
