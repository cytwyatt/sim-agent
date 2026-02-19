from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RankingWeights:
    relevance_weight: float = 0.6
    recency_weight: float = 0.25
    citation_weight: float = 0.15


@dataclass
class DefaultSettings:
    top_n: int = 10
    years: int = 10
    output_dir: str = "outputs"
    deep_profiles: list[str] = field(default_factory=lambda: ["MD"])
    use_sqlite: bool = True


@dataclass
class ModelSettings:
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = "https://api.openai.com/v1"


@dataclass
class ValidationSettings:
    md_timestep_fs_min: float = 0.1
    md_timestep_fs_max: float = 10.0
    temperature_k_min: float = 1.0
    temperature_k_max: float = 2000.0
    pressure_bar_max: float = 10000.0


@dataclass
class AppConfig:
    defaults: DefaultSettings = field(default_factory=DefaultSettings)
    ranking: RankingWeights = field(default_factory=RankingWeights)
    models: ModelSettings = field(default_factory=ModelSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    semantic_scholar_api_key: str | None = None
    openai_api_key: str | None = None


def _as_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else Path("sim_agent.toml")
    _load_dotenv(path.parent / ".env")
    data = _as_dict(path)

    defaults_data = data.get("defaults", {})
    ranking_data = data.get("ranking", {})
    model_data = data.get("models", {})
    validation_data = data.get("validation", {})

    config = AppConfig(
        defaults=DefaultSettings(
            top_n=int(defaults_data.get("top_n", 10)),
            years=int(defaults_data.get("years", 10)),
            output_dir=str(defaults_data.get("output_dir", "outputs")),
            deep_profiles=list(defaults_data.get("deep_profiles", ["MD"])),
            use_sqlite=bool(defaults_data.get("use_sqlite", True)),
        ),
        ranking=RankingWeights(
            relevance_weight=float(ranking_data.get("relevance_weight", 0.6)),
            recency_weight=float(ranking_data.get("recency_weight", 0.25)),
            citation_weight=float(ranking_data.get("citation_weight", 0.15)),
        ),
        models=ModelSettings(
            openai_model=str(model_data.get("openai_model", "gpt-4.1-mini")),
            openai_base_url=str(model_data.get("openai_base_url", "https://api.openai.com/v1")),
        ),
        validation=ValidationSettings(
            md_timestep_fs_min=float(validation_data.get("md_timestep_fs_min", 0.1)),
            md_timestep_fs_max=float(validation_data.get("md_timestep_fs_max", 10.0)),
            temperature_k_min=float(validation_data.get("temperature_k_min", 1.0)),
            temperature_k_max=float(validation_data.get("temperature_k_max", 2000.0)),
            pressure_bar_max=float(validation_data.get("pressure_bar_max", 10000.0)),
        ),
        semantic_scholar_api_key=os.getenv("S2_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return config
