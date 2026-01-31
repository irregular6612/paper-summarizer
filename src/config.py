"""설정 관리 모듈"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    """LLM 설정"""
    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "llama3.2"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    max_sections: int = 50  # 요약할 최대 섹션 수


@dataclass
class PathsConfig:
    """경로 설정"""
    watch_dir: str = "target-pdf"
    output_dir: str = "output"
    processed_dir: str | None = None  # None이면 이동 안함


@dataclass
class MarkdownConfig:
    """마크다운 생성 설정"""
    include_toc: bool = True
    include_frontmatter: bool = True
    include_figures: bool = True
    include_tables: bool = True
    summary_style: str = "blockquote"  # blockquote, callout
    assets_folder: str = "assets"
    default_tags: list[str] = field(default_factory=lambda: ["paper"])


@dataclass
class OutputConfig:
    """출력 설정"""
    image_dpi: int = 150
    summary_language: str = "korean"


@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    markdown: MarkdownConfig = field(default_factory=MarkdownConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _expand_env_vars(value: str) -> str:
    """환경변수를 확장합니다. ${VAR} 또는 $VAR 형식 지원"""
    if isinstance(value, str) and "$" in value:
        return os.path.expandvars(value)
    return value


def _dict_to_config(data: dict) -> AppConfig:
    """딕셔너리를 AppConfig로 변환합니다."""
    llm_data = data.get("llm", {})
    paths_data = data.get("paths", {})
    markdown_data = data.get("markdown", {})
    output_data = data.get("output", {})

    # 환경변수 확장
    if "api_key" in llm_data:
        llm_data["api_key"] = _expand_env_vars(llm_data["api_key"])

    return AppConfig(
        llm=LLMConfig(**llm_data),
        paths=PathsConfig(**paths_data),
        markdown=MarkdownConfig(**markdown_data),
        output=OutputConfig(**output_data),
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    설정 파일을 로드합니다.

    Args:
        config_path: 설정 파일 경로. None이면 기본 위치에서 찾음.

    Returns:
        AppConfig 객체
    """
    # 기본 설정 파일 위치들
    search_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".paper-summary" / "config.yaml",
    ]

    if config_path:
        search_paths.insert(0, Path(config_path))

    # 첫 번째로 찾은 설정 파일 사용
    for path in search_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _dict_to_config(data)

    # 설정 파일이 없으면 기본값 사용
    return AppConfig()


def save_default_config(config_path: str | Path = "config.yaml") -> None:
    """기본 설정 파일을 생성합니다."""
    default_config = """# Paper Summary Pipeline 설정

llm:
  # 사용할 LLM 제공자: ollama, openai, anthropic
  provider: "ollama"
  
  # 모델명
  # - ollama: llama3.2, mistral, gemma2 등
  # - openai: gpt-4o-mini, gpt-4o 등
  # - anthropic: claude-3-haiku-20240307, claude-3-5-sonnet-20241022 등
  model: "llama3.2"
  
  # API 키 (환경변수 참조 가능)
  # api_key: "${OPENAI_API_KEY}"
  
  # Ollama base URL (기본: http://localhost:11434)
  # base_url: "http://localhost:11434"
  
  temperature: 0.3
  max_tokens: 1024

paths:
  # PDF 파일을 넣을 폴더
  watch_dir: "target-pdf"
  
  # 결과물 출력 폴더
  output_dir: "output"
  
  # 처리 완료된 PDF 이동 폴더 (비워두면 이동 안함)
  # processed_dir: "processed"

markdown:
  include_toc: true
  include_frontmatter: true
  include_figures: true
  include_tables: true
  
  # 요약 스타일: blockquote 또는 callout
  summary_style: "blockquote"
  
  assets_folder: "assets"
  default_tags:
    - paper

output:
  # 이미지 해상도 (DPI)
  image_dpi: 150
  
  # 요약 출력 언어
  summary_language: "korean"
"""
    Path(config_path).write_text(default_config, encoding="utf-8")
