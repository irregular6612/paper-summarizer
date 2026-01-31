"""마크다운 생성 모듈 - Obsidian 호환 형식"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .extractor import PaperStructure, Section


@dataclass
class MarkdownConfig:
    """마크다운 생성 설정"""
    include_toc: bool = True
    include_frontmatter: bool = True
    include_figures: bool = True
    include_tables: bool = True
    summary_style: str = "blockquote"  # blockquote, callout
    assets_folder: str = "assets"
    default_tags: list[str] | None = None


class MarkdownGenerator:
    """Obsidian 호환 마크다운을 생성하는 클래스"""

    def __init__(self, config: MarkdownConfig | None = None):
        self.config = config or MarkdownConfig()

    def generate(
        self,
        structure: PaperStructure,
        summaries: dict[str, str],
        image_paths: dict[str, Path],
    ) -> str:
        """
        논문 구조, 요약, 이미지 경로를 기반으로 마크다운을 생성합니다.

        Args:
            structure: 논문 구조
            summaries: {섹션 제목: 요약} 매핑
            image_paths: {figure/table id: 이미지 경로} 매핑

        Returns:
            Obsidian 호환 마크다운 문자열
        """
        parts = []

        # 1. Frontmatter
        if self.config.include_frontmatter:
            parts.append(self._generate_frontmatter(structure, summaries))

        # 2. 제목
        parts.append(f"# {structure.title}\n")

        # 3. 전체 요약 (있는 경우)
        if "overview" in summaries:
            parts.append(self._format_summary("Overview", summaries["overview"]))
            parts.append("")

        # 4. 목차
        if self.config.include_toc:
            parts.append(self._generate_toc(structure))

        # 5. 각 섹션
        for section in structure.sections:
            parts.append(self._generate_section(section, summaries, image_paths))

        return "\n".join(parts)

    def _generate_frontmatter(
        self,
        structure: PaperStructure,
        summaries: dict[str, str],
    ) -> str:
        """YAML frontmatter를 생성합니다."""
        tags = self.config.default_tags or ["paper"]

        # 제목에서 약어 추출 시도 (괄호 안의 내용)
        aliases = []
        title = structure.title
        if "(" in title and ")" in title:
            start = title.rfind("(")
            end = title.rfind(")")
            if start < end:
                alias = title[start + 1 : end].strip()
                if alias and len(alias) < 20:
                    aliases.append(alias)

        lines = [
            "---",
            f'title: "{structure.title}"',
        ]

        if aliases:
            lines.append(f'aliases: {aliases}')

        lines.append(f"tags: {tags}")
        lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _generate_toc(self, structure: PaperStructure) -> str:
        """목차를 생성합니다."""
        lines = ["## 목차", ""]

        for section in structure.sections:
            # 섹션 제목을 Obsidian 내부 링크로 변환
            anchor = section.title.replace(" ", " ")
            indent = "  " * (section.level - 1) if section.level > 1 else ""
            lines.append(f"{indent}- [[#{section.title}|{section.title}]]")

        lines.append("")
        return "\n".join(lines)

    def _generate_section(
        self,
        section: Section,
        summaries: dict[str, str],
        image_paths: dict[str, Path],
    ) -> str:
        """단일 섹션의 마크다운을 생성합니다."""
        lines = []

        # 섹션 제목 (레벨에 따라 # 개수 조정)
        header_level = min(section.level + 1, 6)  # 최대 h6까지
        lines.append(f"{'#' * header_level} {section.title}")
        lines.append("")

        # 요약 (있는 경우)
        if section.title in summaries:
            lines.append(self._format_summary("요약", summaries[section.title]))
            lines.append("")

        # Figure 이미지
        if self.config.include_figures:
            for fig in section.figures:
                if fig.id in image_paths:
                    img_name = image_paths[fig.id].name
                    lines.append(f"![[{self.config.assets_folder}/{img_name}]]")
                    if fig.caption:
                        lines.append(f"*{fig.caption}*")
                    lines.append("")

        # Table
        if self.config.include_tables:
            for table in section.tables:
                if table.id in image_paths:
                    # 이미지로 표시
                    img_name = image_paths[table.id].name
                    lines.append(f"![[{self.config.assets_folder}/{img_name}]]")
                    if table.caption:
                        lines.append(f"*{table.caption}*")
                elif table.markdown:
                    # 마크다운 테이블로 표시
                    if table.caption:
                        lines.append(f"*{table.caption}*")
                    lines.append(table.markdown)
                lines.append("")

        return "\n".join(lines)

    def _format_summary(self, label: str, summary: str) -> str:
        """요약을 포맷팅합니다."""
        if self.config.summary_style == "callout":
            # Obsidian callout 스타일
            lines = [f"> [!note] {label}"]
            for line in summary.split("\n"):
                lines.append(f"> {line}")
            return "\n".join(lines)
        else:
            # 기본 blockquote 스타일
            lines = [f"> **[{label}]**"]
            for line in summary.split("\n"):
                lines.append(f"> {line}")
            return "\n".join(lines)


def generate_markdown(
    structure: PaperStructure,
    summaries: dict[str, str],
    image_paths: dict[str, Path],
    config: MarkdownConfig | None = None,
) -> str:
    """마크다운을 생성하는 편의 함수"""
    generator = MarkdownGenerator(config)
    return generator.generate(structure, summaries, image_paths)


def save_markdown(
    content: str,
    output_path: Path,
) -> Path:
    """마크다운을 파일로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
