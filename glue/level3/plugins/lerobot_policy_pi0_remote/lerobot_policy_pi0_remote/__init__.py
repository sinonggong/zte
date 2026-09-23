"""LeRobot plugin: `pi0_remote` policy type.

Installed as a `lerobot_policy_*` distribution so `register_third_party_plugins()`
imports it and `--policy.type=pi0_remote` / a checkpoint dir whose config.json says
`"type": "pi0_remote"` resolve to Pi0RemotePolicy.  Nothing in LeRobot is edited.
"""

from .configuration_pi0_remote import Pi0RemoteConfig  # noqa: F401  (registers the type)
from .modeling_pi0_remote import Pi0RemotePolicy  # noqa: F401

__all__ = ["Pi0RemoteConfig", "Pi0RemotePolicy"]
