"""이미지 추출 모듈 - marker-pdf 이미지 저장"""

import re
from pathlib import Path

from .extractor import PaperStructure


def sanitize_filename(name: str, max_length: int = 30) -> str:
    """파일명으로 사용할 수 있도록 정리합니다."""
    # 파일명에 사용할 수 없는 문자 제거
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "")

    # 공백을 언더스코어로 변환
    name = re.sub(r"\s+", "_", name)

    # 연속된 언더스코어 제거
    name = re.sub(r"_+", "_", name)

    # 길이 제한
    if len(name) > max_length:
        name = name[:max_length]

    return name.strip("_")


def create_short_paper_name(title: str, pdf_filename: str = "") -> str:
    """논문 제목 또는 파일명에서 짧은 이름을 생성합니다."""
    # 1. 제목에서 괄호 안의 약어 찾기 (예: CLIP, GPT 등)
    match = re.search(r"\(([A-Z]{2,10})\)", title)
    if match:
        return match.group(1)

    # 2. PDF 파일명에서 약어 찾기
    if pdf_filename:
        match = re.search(r"\(([A-Z]{2,10})\)", pdf_filename)
        if match:
            return match.group(1)

    # 3. 없으면 첫 몇 단어 사용
    words = title.split()[:3]
    return sanitize_filename("_".join(words), max_length=20)


class ImageExporter:
    """marker에서 추출한 이미지를 저장하는 클래스"""

    def __init__(self, paper_name_prefix: bool = True):
        """
        Args:
            paper_name_prefix: 파일명에 논문명 프리픽스 추가 여부
        """
        self.paper_name_prefix = paper_name_prefix

    def export_all(
        self,
        structure: PaperStructure,
        output_dir: Path,
        pdf_filename: str = "",
    ) -> dict[str, Path]:
        """
        모든 이미지를 저장합니다.

        Args:
            structure: 논문 구조
            output_dir: 출력 디렉토리
            pdf_filename: PDF 파일명 (약어 추출용)

        Returns:
            dict: {원본 id: 저장된 이미지 경로} 매핑
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = {}

        # 논문명 프리픽스 생성
        prefix = ""
        if self.paper_name_prefix:
            prefix = create_short_paper_name(structure.title, pdf_filename) + "_"

        # marker가 추출한 이미지 저장
        count = 0
        for img_name, img_data in structure.images.items():
            if img_data:
                count += 1
                # 파일 확장자 확인
                ext = Path(img_name).suffix or ".png"
                if not ext.startswith("."):
                    ext = "." + ext

                # Figure인지 Table인지 구분
                is_table = "table" in img_name.lower()
                img_type = "table" if is_table else "fig"

                new_name = f"{prefix}{img_type}{count}{ext}"
                img_path = output_dir / new_name

                try:
                    img_path.write_bytes(img_data)
                    exported[img_name] = img_path
                except Exception:
                    pass

        return exported


def export_images(
    structure: PaperStructure,
    output_dir: str | Path,
    paper_name_prefix: bool = True,
) -> dict[str, Path]:
    """이미지를 추출하는 편의 함수"""
    exporter = ImageExporter(paper_name_prefix=paper_name_prefix)
    return exporter.export_all(structure, Path(output_dir))
