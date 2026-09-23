from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("pi0_remote")
@dataclass
class Pi0RemoteConfig(PreTrainedConfig):
    """A pi0-shaped policy whose predict_action_chunk runs on a remote host.

    Everything the robot side needs (feature layout, chunking, normalisation
    modes, tokenizer length) mirrors the real pi0 checkpoint so the stock
    pre/post-processors saved next to this config keep working unchanged.
    """

    # where the chunks come from
    server_url: str = "http://192.168.10.1:8081"
    request_timeout_s: float = 180.0
    remote_policy_type: str = "pi0"     # informational: what the server is serving
    source_checkpoint: str | None = None  # informational: the checkpoint whose processors were copied
    dtype: str = "float32"   # accepted for CLI parity with pi0 (--policy.dtype); nothing to cast here

    # pi0 layout (copied from the checkpoint's config.json by make_pi0_remote_policy_dir.py)
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenizer_max_length: int = 48
    image_resolution: tuple[int, int] = (224, 224)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    action_feature_names: list[str] | None = None
    rtc_config: RTCConfig | None = None   # set by lerobot_rollout in RTC mode; forwarded to the server

    def __post_init__(self):
        super().__post_init__()
        self.image_resolution = tuple(self.image_resolution)

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            raise ValueError(f"{OBS_STATE} must be an input feature")
        if ACTION not in self.output_features:
            raise ValueError(f"{ACTION} must be an output feature")
        for key, ft in self.input_features.items():
            if ft.type is FeatureType.VISUAL and tuple(ft.shape[-2:]) != tuple(self.image_resolution):
                raise ValueError(f"{key} shape {ft.shape} does not match image_resolution {self.image_resolution}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig()

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


__all__ = ["Pi0RemoteConfig", "PolicyFeature"]
