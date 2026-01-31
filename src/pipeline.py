"""파이프라인 모듈 - 전체 처리 흐름 통합 (marker-pdf 기반)"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import AppConfig
from .extractor import get_extractor, PaperStructure, strip_html_tags
from .image_export import ImageExporter, create_short_paper_name, sanitize_filename
from .summarizer import Summarizer, SummaryConfig
from .markdown_gen import (
    MarkdownGenerator,
    MarkdownConfig as MDConfig,
    save_markdown,
)

logger = logging.getLogger(__name__)


class PaperPipeline:
    """논문 PDF를 처리하는 전체 파이프라인 (marker-pdf 기반)"""

    def __init__(self, config: AppConfig):
        self.config = config
        # 싱글톤 extractor 사용 (모델 재사용으로 성능 향상)
        self.extractor = get_extractor(
            use_llm=getattr(config.llm, "use_marker_llm", False),
            force_ocr=getattr(config.output, "force_ocr", False),
        )
        self.image_exporter = ImageExporter(paper_name_prefix=True)
        self.summarizer = self._create_summarizer()
        self.markdown_gen = self._create_markdown_gen()

    def _create_summarizer(self) -> Summarizer:
        """Summarizer 인스턴스를 생성합니다."""
        return Summarizer(
            SummaryConfig(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                language=self.config.output.summary_language,
                max_sections=getattr(self.config.llm, 'max_sections', 999),
            )
        )

    def _create_markdown_gen(self) -> MarkdownGenerator:
        """MarkdownGenerator 인스턴스를 생성합니다."""
        return MarkdownGenerator(
            MDConfig(
                include_toc=self.config.markdown.include_toc,
                include_frontmatter=self.config.markdown.include_frontmatter,
                include_figures=self.config.markdown.include_figures,
                include_tables=self.config.markdown.include_tables,
                summary_style=self.config.markdown.summary_style,
                assets_folder=self.config.markdown.assets_folder,
                default_tags=self.config.markdown.default_tags,
            )
        )

    def process(self, pdf_path: Path) -> Path:
        """
        PDF 논문을 처리하여 Obsidian 마크다운을 생성합니다.

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            생성된 마크다운 파일 경로
        """
        logger.info(f"1/4 Extracting document structure: {pdf_path.name}")

        # 1. PDF 구조 추출 (marker-pdf 사용)
        structure = self.extractor.extract(pdf_path)
        logger.info(
            f"     Found {len(structure.sections)} sections, "
            f"{len(structure.figures)} figures, {len(structure.tables)} tables, "
            f"{len(structure.images)} images"
        )

        # HTML 태그 제거 (marker-pdf가 생성하는 <span> 등)
        structure.markdown = strip_html_tags(structure.markdown) if structure.markdown else ""

        # 출력 디렉토리 설정
        paper_name = self._sanitize_filename(structure.title or pdf_path.stem)
        output_base = Path(self.config.paths.output_dir) / paper_name
        assets_dir = output_base / self.config.markdown.assets_folder

        # 2&3. 이미지 저장 + LLM 요약 병렬 실행
        logger.info("2/4 Exporting figures and tables + 3/4 Generating summaries (parallel)...")
        
        def export_images():
            return self.image_exporter.export_all(
                structure, assets_dir, pdf_filename=pdf_path.name
            )
        
        def generate_summaries():
            try:
                return self.summarizer.summarize_paper(structure)
            except Exception as e:
                logger.warning(f"     Summarization failed: {e}")
                logger.warning("     Continuing without summaries...")
                return {}

        def extract_metadata():
            try:
                return self.summarizer.extract_metadata(structure)
            except Exception as e:
                logger.warning(f"     Metadata extraction failed: {e}")
                return {"authors": [], "journal": None, "year": None}

        with ThreadPoolExecutor(max_workers=3) as executor:
            image_future = executor.submit(export_images)
            summary_future = executor.submit(generate_summaries)
            metadata_future = executor.submit(extract_metadata)

            image_paths = image_future.result()
            summaries = summary_future.result()
            metadata = metadata_future.result()
        
        logger.info(f"     Exported {len(image_paths)} images")
        logger.info(f"     Generated {len(summaries)} summaries")

        # 4. 마크다운 생성
        logger.info("4/4 Generating Obsidian markdown...")
        markdown_content = self._generate_enhanced_markdown(
            structure, summaries, image_paths, metadata
        )

        md_path = output_base / f"{paper_name}.md"
        save_markdown(markdown_content, md_path)
        logger.info(f"     Saved to: {md_path}")

        return md_path

    def _generate_enhanced_markdown(
        self,
        structure: PaperStructure,
        summaries: dict[str, str],
        image_paths: dict[str, Path],
        metadata: dict | None = None,
    ) -> str:
        """marker의 마크다운에 요약과 메타데이터를 추가합니다."""
        parts = []

        # 1. Frontmatter
        if self.config.markdown.include_frontmatter:
            parts.append(self._generate_frontmatter(structure, metadata))

        # 2. 제목
        parts.append(f"# {structure.title}\n")

        # 3. 전체 요약 (있는 경우)
        if "overview" in summaries:
            parts.append(self._format_summary("summary", summaries["overview"], "Overview"))
            parts.append("")

        # 4. 목차
        if self.config.markdown.include_toc and structure.sections:
            parts.append(self._generate_toc(structure))

        # 5. 이미지 링크 교체 (원본 이름 → 새 파일명)
        enhanced_md = self._replace_image_links(structure.markdown, image_paths)

        # 6. 섹션별 요약 추가
        enhanced_md = self._add_summaries_to_markdown(enhanced_md, summaries, structure)

        # 7. References 섹션 제거
        enhanced_md = self._remove_references_section(enhanced_md)

        parts.append(enhanced_md)

        return "\n".join(parts)

    def _generate_frontmatter(self, structure: PaperStructure, metadata: dict | None = None) -> str:
        """YAML frontmatter를 생성합니다 (review-template 형식)."""
        from datetime import datetime

        tags = self.config.markdown.default_tags or ["paper"]
        metadata = metadata or {}

        # 제목에서 약어 추출 시도
        aliases = []
        short_name = create_short_paper_name(structure.title)
        if short_name and short_name != structure.title:
            aliases.append(short_name)

        # 저자 목록
        authors = metadata.get("authors") or []
        journal = metadata.get("journal") or ""
        year = metadata.get("year") or ""

        lines = [
            "---",
            "Reading-Status: ",
        ]

        # Author
        if authors:
            lines.append("Author:")
            for author in authors:
                lines.append(f"  - {author}")
        else:
            lines.append("Author: ")

        lines.append(f'Journal: "{journal}"')
        lines.append(f"Published Year: {year}")
        lines.append("Topic: ")
        lines.append(f"Review-Date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("Comment: ")
        lines.append("URL: ")
        lines.append("isTargetPaper: ")
        lines.append('linked-bases: ""')
        lines.append(f'title: "{structure.title}"')

        if aliases:
            lines.append(f"aliases: {aliases}")

        lines.append(f"tags: {tags}")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _generate_toc(self, structure: PaperStructure) -> str:
        """목차를 생성합니다."""
        lines = ["## 목차", ""]

        for section in structure.sections:
            indent = "  " * (section.level - 1) if section.level > 1 else ""
            lines.append(f"{indent}- [[#{section.title}|{section.title}]]")

        lines.append("")
        return "\n".join(lines)

    def _replace_image_links(
        self,
        markdown: str,
        image_paths: dict[str, Path],
    ) -> str:
        """마크다운의 이미지 링크를 새 파일명으로 교체합니다."""
        import re

        result = markdown
        assets_folder = self.config.markdown.assets_folder

        for original_name, new_path in image_paths.items():
            new_name = f"{assets_folder}/{new_path.name}"
            # replacement 문자열에서 백슬래시 이스케이프 (re.sub에서 \가 특수 의미를 가짐)
            safe_replacement = f"![[{new_name}]]".replace("\\", "\\\\")

            # ![](old) → ![[assets/new]] (Obsidian 형식)
            pattern = rf"!\[\]\({re.escape(original_name)}\)"
            result = re.sub(pattern, safe_replacement, result)

            # ![alt](old) → ![[assets/new]] (alt text 포함 케이스)
            pattern_with_alt = rf"!\[[^\]]*\]\({re.escape(original_name)}\)"
            result = re.sub(pattern_with_alt, safe_replacement, result)

        return result

    def _add_summaries_to_markdown(
        self,
        markdown: str,
        summaries: dict[str, str],
        structure: PaperStructure,
    ) -> str:
        """마크다운에 섹션별 요약을 추가합니다."""
        import re
        
        if not summaries:
            return markdown

        result = markdown
        inserted_count = 0

        # 각 섹션 제목 뒤에 요약 추가
        for section in structure.sections:
            if section.title in summaries:
                summary_block = self._format_summary("note", summaries[section.title], "Summary")
                # replacement 문자열에서 백슬래시 이스케이프 (re.sub에서 \가 특수 의미를 가짐)
                safe_summary = summary_block.replace("\\", "\\\\")
                # 섹션 제목 찾아서 요약 추가
                # ## 섹션제목 또는 # 섹션제목 패턴 찾기
                # f-string에서 {1,6}이 튜플로 해석되는 버그 수정
                escaped_title = re.escape(section.title)
                pattern = r"(#{1,6}\s+" + escaped_title + r"\s*\n)"
                new_result = re.sub(pattern, r"\1\n" + safe_summary + r"\n\n", result, count=1)
                
                if new_result != result:
                    inserted_count += 1
                    result = new_result

        logger.info(f"     Inserted {inserted_count} section summaries")
        return result

    def _format_summary(self, callout_type: str, summary: str, title: str = "") -> str:
        """요약을 포맷팅합니다.

        Args:
            callout_type: Obsidian callout type (note, summary, important, warning, question)
            summary: 요약 내용
            title: callout 제목 (optional)
        """
        header = f"> [!{callout_type}] {title}" if title else f"> [!{callout_type}]"
        lines = [header]
        for line in summary.split("\n"):
            lines.append(f"> {line}")
        return "\n".join(lines)

    def _remove_references_section(self, markdown: str) -> str:
        """References/Bibliography 섹션과 그 이후 내용을 제거합니다."""
        import re
        pattern = r"\n#{1,6}\s+(References|Bibliography|참고\s*문헌)\s*\n.*"
        return re.sub(pattern, "", markdown, flags=re.DOTALL | re.IGNORECASE)

    def _sanitize_filename(self, name: str) -> str:
        """파일명으로 사용할 수 없는 문자를 제거합니다."""
        return sanitize_filename(name, max_length=100)


def process_paper(pdf_path: str | Path, config: AppConfig | None = None) -> Path:
    """논문을 처리하는 편의 함수"""
    if config is None:
        from .config import load_config
        config = load_config()

    pipeline = PaperPipeline(config)
    return pipeline.process(Path(pdf_path))
