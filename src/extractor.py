"""PDF 파싱 모듈 - marker-pdf 기반 문서 구조 추출"""

import io
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# M1/M2 Mac에서 MPS 사용 활성화 (marker-pdf 로드 전에 설정)
# TableRecEncoderDecoderModel은 MPS 미지원으로 자동 CPU 폴백됨
if not os.environ.get("TORCH_DEVICE"):
    os.environ["TORCH_DEVICE"] = "mps"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """논문 섹션 정보"""
    title: str
    level: int
    content: str = ""
    figures: list["FigureInfo"] = field(default_factory=list)
    tables: list["TableInfo"] = field(default_factory=list)


@dataclass
class FigureInfo:
    """Figure 정보"""
    id: str
    caption: str
    image_data: bytes | None = None


@dataclass
class TableInfo:
    """Table 정보"""
    id: str
    caption: str
    markdown: str = ""
    image_data: bytes | None = None


@dataclass
class PaperStructure:
    """논문 전체 구조"""
    title: str
    markdown: str
    sections: list[Section]
    figures: list[FigureInfo]
    tables: list[TableInfo]
    images: dict[str, bytes]
    metadata: dict[str, Any] = field(default_factory=dict)


class PDFExtractor:
    """PDF에서 문서 구조를 추출하는 클래스 (marker-pdf 기반)"""

    def __init__(self, use_llm: bool = False, force_ocr: bool = False):
        """
        Args:
            use_llm: LLM을 사용하여 정확도 향상 (기본 False)
            force_ocr: 모든 페이지에 OCR 강제 적용 (기본 False)
        """
        self.use_llm = use_llm
        self.force_ocr = force_ocr
        self._converter = None
        self._model_dict = None

    def _get_converter(self) -> PdfConverter:
        """Converter를 lazy하게 생성합니다."""
        if self._converter is None:
            if self._model_dict is None:
                self._model_dict = create_model_dict()

            config = {
                "output_format": "markdown",
            }
            if self.force_ocr:
                config["force_ocr"] = True

            config_parser = ConfigParser(config)

            self._converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=self._model_dict,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
            )

        return self._converter

    def extract(self, pdf_path: Path) -> PaperStructure:
        """PDF에서 문서 구조를 추출합니다."""
        converter = self._get_converter()

        # PDF 변환 실행
        rendered = converter(str(pdf_path))

        # 마크다운 텍스트와 이미지 추출
        markdown_text, _, images = text_from_rendered(rendered)

        # 제목 추출
        title = self._extract_title_from_markdown(markdown_text, pdf_path)

        # 마크다운에서 섹션 추출
        sections = self._extract_sections_from_markdown(markdown_text)

        # Figure, Table 카운트
        figures = self._extract_figures_from_markdown(markdown_text)
        tables = self._extract_tables_from_markdown(markdown_text)

        # images는 PIL Image 또는 bytes일 수 있음
        images_bytes = self._convert_images_to_bytes(images)

        return PaperStructure(
            title=title,
            markdown=markdown_text,
            sections=sections,
            figures=figures,
            tables=tables,
            images=images_bytes,
            metadata=rendered.metadata if hasattr(rendered, "metadata") else {},
        )

    def _convert_images_to_bytes(self, images: dict) -> dict[str, bytes]:
        """이미지를 bytes로 변환합니다."""
        result = {}

        logger.debug(f"Converting {len(images)} images to bytes")

        for name, img in images.items():
            try:
                if isinstance(img, bytes):
                    result[name] = img
                    logger.debug(f"  {name}: already bytes ({len(img)} bytes)")
                elif hasattr(img, "save"):
                    # PIL Image인 경우
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    data = buffer.getvalue()
                    result[name] = data
                    logger.debug(f"  {name}: PIL Image converted ({len(data)} bytes)")
                elif hasattr(img, "tobytes"):
                    # numpy array 등
                    result[name] = img.tobytes()
                    logger.debug(f"  {name}: converted via tobytes()")
                else:
                    logger.warning(f"  {name}: unknown type {type(img)}, skipping")
            except Exception as e:
                logger.warning(f"  {name}: conversion failed - {e}")

        return result

    def _extract_title_from_markdown(self, markdown: str, pdf_path: Path) -> str:
        """마크다운에서 제목을 추출합니다."""
        # 첫 번째 # 헤더 찾기
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # 실패시 파일명에서 추출
        return pdf_path.stem

    def _extract_sections_from_markdown(self, markdown: str) -> list[Section]:
        """마크다운에서 섹션을 추출합니다."""
        sections = []
        pattern = r"^(#{1,6})\s+(.+)$"

        for match in re.finditer(pattern, markdown, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append(Section(title=title, level=level))

        return sections

    def _extract_figures_from_markdown(self, markdown: str) -> list[FigureInfo]:
        """마크다운에서 Figure 정보를 추출합니다."""
        figures = []
        # 이미지 패턴: ![alt](path) 또는 ![[path]]
        pattern = r"!\[([^\]]*)\]\(([^)]+)\)|!\[\[([^\]]+)\]\]"

        count = 0
        for match in re.finditer(pattern, markdown):
            count += 1
            caption = match.group(1) or match.group(3) or ""

            figures.append(FigureInfo(
                id=f"figure{count}",
                caption=caption,
                image_data=None,
            ))

        return figures

    def _extract_tables_from_markdown(self, markdown: str) -> list[TableInfo]:
        """마크다운에서 Table 정보를 추출합니다."""
        tables = []

        # 마크다운 테이블 블록 카운트 (| 로 시작하는 행들)
        table_pattern = r"(\|[^\n]+\|\n)+"
        table_blocks = re.findall(table_pattern, markdown)

        for i in range(len(table_blocks)):
            tables.append(TableInfo(
                id=f"table{i + 1}",
                caption=f"Table {i + 1}",
                markdown=table_blocks[i] if i < len(table_blocks) else "",
                image_data=None,
            ))

        return tables


# 모듈 레벨 싱글톤 (모델 재사용으로 성능 향상)
_global_extractor: PDFExtractor | None = None


def get_extractor(use_llm: bool = False, force_ocr: bool = False) -> PDFExtractor:
    """
    싱글톤 PDFExtractor를 반환합니다.
    
    모델을 한 번만 로드하고 재사용하여 여러 PDF 처리 시 성능을 향상시킵니다.
    
    Args:
        use_llm: LLM을 사용하여 정확도 향상 (첫 호출 시에만 적용)
        force_ocr: 모든 페이지에 OCR 강제 적용 (첫 호출 시에만 적용)
    
    Returns:
        PDFExtractor 싱글톤 인스턴스
    """
    global _global_extractor
    if _global_extractor is None:
        logger.info("Creating PDFExtractor singleton...")
        _global_extractor = PDFExtractor(use_llm=use_llm, force_ocr=force_ocr)
    return _global_extractor


def warmup_extractor() -> None:
    """
    모델을 미리 로드합니다.
    
    애플리케이션 시작 시 호출하면 첫 PDF 처리 시 대기 시간을 줄일 수 있습니다.
    """
    logger.info("Warming up PDF extractor models...")
    extractor = get_extractor()
    extractor._get_converter()  # 모델 로드 트리거
    logger.info("PDF extractor models ready!")


def extract_paper(pdf_path: str | Path, use_llm: bool = False) -> PaperStructure:
    """PDF 논문에서 구조를 추출하는 편의 함수"""
    extractor = get_extractor(use_llm=use_llm)
    return extractor.extract(Path(pdf_path))
