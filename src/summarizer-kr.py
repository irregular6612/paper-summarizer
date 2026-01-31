"""LLM 요약 모듈 - litellm 기반 다중 LLM 지원"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import litellm
from litellm import completion

from .extractor import Section, PaperStructure

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """요약 설정"""
    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "qwen3:14b"
    api_key: str | None = None
    base_url: str | None = None  # Ollama의 경우 http://localhost:11434
    temperature: float = 0.6  # Qwen3 thinking mode 권장값
    max_tokens: int = 2048  # 요약에 충분한 크기
    language: str = "korean"  # 요약 출력 언어
    max_sections: int = 50  # 요약할 최대 섹션 수


# 프롬프트 템플릿
SECTION_SUMMARY_PROMPT = """# 작업
아래 논문 섹션을 **반드시 {language}로만** 구조화하여 요약하세요.
{context}
# 섹션 정보
- 제목: {title}
- 내용:
{content}

# 출력 요구사항
- 언어: **{language}** (영어로 작성 금지, 기술 용어만 영어 허용)
- 반드시 bullet point(`-`) 형식으로 작성
- 다음 구조를 따르되, 해당 없는 항목은 생략:
  (1) 이 섹션의 핵심 목적/주장
  (2) 제안된 방법/접근법
  (3) 주요 결과/의의 (수치, 벤치마크가 있으면 반드시 포함)
- 내용이 짧은 섹션은 1-5개 bullet point로 간결하게
- 내용이 긴 섹션은 최대 5-10개 bullet point
- 마크다운 heading(#, ##, ### 등) 사용 금지
- 이전 섹션과의 연결성을 고려하여 요약"""

MARKDOWN_SUMMARY_PROMPT = """# 작업
논문 마크다운의 각 주요 섹션을 {language}로 2-3문장씩 요약

# 논문 마크다운
{markdown}

# 출력 형식
## 섹션제목
요약 내용

## 다음섹션
요약 내용

# {language} 요약"""

PAPER_OVERVIEW_PROMPT = """# 작업
논문 전체의 핵심을 **반드시 {language}로만** bullet point 5-10개로 요약하세요.

# 논문 정보
- 제목: {title}
- 섹션별 요약:
{section_summaries}

# 출력 요구사항
- 언어: **{language}** (영어로 작성 금지, 기술 용어만 영어 허용)
- 반드시 bullet point(`-`) 형식으로 5-10개 작성
- 마크다운 heading(#, ##, ### 등) 절대 사용 금지
- 다음 순서로 구성:
  (1) 연구 동기/배경 (1개)
  (2) 핵심 방법론 (1-2개)
  (3) 주요 기여점 (1-2개)
  (4) 핵심 실험 결과 — 수치/벤치마크 반드시 1개 이상 포함 (1개)
  (5) 의의 또는 한계 (1개)
- 수식이 있으면 원문 LaTeX 표기 유지 (예: $\\alpha$, $L_{{total}}$)"""

METADATA_EXTRACT_PROMPT = """# 작업
아래 논문 텍스트에서 메타데이터를 추출하세요.

# 논문 텍스트 (앞부분)
{text}

# 출력 요구사항
- 반드시 아래 JSON 형식으로만 출력 (다른 텍스트 없이)
- 추출할 수 없는 항목은 null로 표기
```json
{{"authors": ["저자1", "저자2", ...], "journal": "학회/저널명", "year": "출판연도"}}
```"""


litellm.set_verbose = True # for debugging.

class Summarizer:
    """LLM을 사용하여 논문을 요약하는 클래스"""

    def __init__(self, config: SummaryConfig):
        self.config = config
        self._setup_litellm()

    def _setup_litellm(self) -> None:
        """LiteLLM 설정을 초기화합니다."""
        if self.config.api_key:
            if self.config.provider == "openai":
                litellm.openai_key = self.config.api_key
            elif self.config.provider == "anthropic":
                litellm.anthropic_key = self.config.api_key

    def _get_model_name(self) -> str:
        """LiteLLM에서 사용할 모델 이름을 반환합니다."""
        provider = self.config.provider
        model = self.config.model

        if provider == "ollama":
            return f"ollama/{model}"
        elif provider == "openai":
            return model
        elif provider == "anthropic":
            return model
        else:
            return model

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트를 반환합니다."""
        return f"""# 역할
당신은 ML/AI/CS 분야 학술 논문 요약 전문가입니다. 딥러닝, 강화학습, 컴퓨터 비전, NLP 등의 논문을 정확하게 분석하고 요약합니다.

# 중요 규칙 (반드시 준수)
- **출력 언어: 반드시 {self.config.language}로만 작성** (영어로 작성 금지)
- 기술 용어(CLIP, PPO, Transformer, attention, backbone 등)만 영어 허용
- 핵심 내용만 간결하게 요약
- 인사말이나 설명 없이 바로 요약 시작

# 출력 형식
- bullet point(`-`) 형식 사용
- 마크다운 heading(#, ##, ### 등) 사용 금지 — Obsidian 문서 구조와 충돌
- 수식/알고리즘 표기 시 원문 LaTeX 유지 (예: $\\alpha$, $L_{{total}}$)
- 모든 문장을 {self.config.language}로 작성"""

    def _call_llm(self, prompt: str) -> str:
        """LLM을 호출하여 응답을 받습니다."""
        model_name = self._get_model_name()
        system_prompt = self._get_system_prompt()

        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.provider == "ollama":
            kwargs["api_base"] = self.config.base_url or "http://localhost:11434"

        response = completion(**kwargs)
        return response.choices[0].message.content

    def summarize_section(self, section: Section, prior_context: str = "") -> str:
        """단일 섹션을 요약합니다."""
        if not section.content.strip():
            return ""

        if len(section.content) < 100:
            return section.content

        context_block = ""
        if prior_context:
            context_block = f"\n# 이전 섹션 요약\n{prior_context}\n"

        prompt = SECTION_SUMMARY_PROMPT.format(
            title=section.title,
            content=section.content[:8000],
            language=self.config.language,
            context=context_block,
        )

        return self._call_llm(prompt)

    def extract_metadata(self, structure: PaperStructure) -> dict:
        """논문에서 저자, 학회, 출판연도 등 메타데이터를 추출합니다."""
        text = (structure.markdown or "")[:2000]
        if not text.strip():
            return {"authors": [], "journal": None, "year": None}

        prompt = METADATA_EXTRACT_PROMPT.format(text=text)

        try:
            raw = self._clean_thinking_tokens(self._call_llm(prompt))
            # JSON 블록 추출
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
                return {
                    "authors": metadata.get("authors") or [],
                    "journal": metadata.get("journal"),
                    "year": metadata.get("year"),
                }
        except Exception as e:
            logger.warning(f"     Metadata extraction failed: {e}")

        return {"authors": [], "journal": None, "year": None}

    def summarize_paper(self, structure: PaperStructure) -> dict[str, str]:
        """논문 전체를 요약합니다."""
        summaries = {}

        if hasattr(structure, "markdown") and structure.markdown:
            summaries = self._summarize_from_markdown(structure)
        else:
            section_summary_texts = []
            for section in structure.sections:
                summary = self.summarize_section(section)
                if summary:
                    summaries[section.title] = summary
                    section_summary_texts.append(f"## {section.title}\n{summary}")

            if section_summary_texts:
                overview_prompt = PAPER_OVERVIEW_PROMPT.format(
                    title=structure.title,
                    section_summaries="\n\n".join(section_summary_texts),
                    language=self.config.language,
                )
                summaries["overview"] = self._call_llm(overview_prompt)

        return summaries

    def _summarize_from_markdown(self, structure: PaperStructure) -> dict[str, str]:
        """마크다운에서 전체 논문을 요약합니다."""
        summaries = {}

        markdown = structure.markdown[:20000]

        # 전체 요약 생성
        logger.info("     Generating overview summary...")
        overview_prompt = PAPER_OVERVIEW_PROMPT.format(
            title=structure.title,
            section_summaries=markdown,
            language=self.config.language,
        )

        summaries["overview"] = self._clean_thinking_tokens(self._call_llm(overview_prompt))

        # 주요 섹션별 요약 - 병렬 처리
        if structure.sections:
            main_sections = [
                s for s in structure.sections
                if s.level <= 4 and self._is_summarizable_section(s.title)
            ]

            max_sections = getattr(self.config, 'max_sections', 15)

            sections_to_summarize = []
            for section in main_sections[:max_sections]:
                section_content = self._extract_section_content(
                    structure.markdown, section.title
                )
                if section_content and len(section_content) > 150:
                    sections_to_summarize.append((section, section_content))

            # 배치별 병렬 처리
            def summarize_single_section(args, prior_context=""):
                section, content = args
                context_block = ""
                if prior_context:
                    context_block = f"\n# 이전 섹션 요약\n{prior_context}\n"
                prompt = SECTION_SUMMARY_PROMPT.format(
                    title=section.title,
                    content=content[:6000],
                    language=self.config.language,
                    context=context_block,
                )
                result = self._call_llm(prompt)
                return section.title, self._clean_thinking_tokens(result)

            if sections_to_summarize:
                batch_size = 3
                accumulated_context = ""

                for i in range(0, len(sections_to_summarize), batch_size):
                    batch = sections_to_summarize[i:i + batch_size]
                    logger.info(f"     Summarizing batch {i // batch_size + 1} ({len(batch)} sections)...")

                    context_for_batch = accumulated_context

                    with ThreadPoolExecutor(max_workers=3) as executor:
                        futures = {
                            executor.submit(summarize_single_section, args, context_for_batch): args[0].title
                            for args in batch
                        }

                        batch_results = []
                        for future in as_completed(futures):
                            title = futures[future]
                            try:
                                section_title, summary = future.result()
                                summaries[section_title] = summary
                                batch_results.append((section_title, summary))
                                logger.info(f"     Completed: {section_title}")
                            except Exception as e:
                                logger.warning(f"     Failed to summarize {title}: {e}")

                    for sec_title, sec_summary in batch_results:
                        sentences = sec_summary.replace('\n', ' ').split('. ')
                        brief = '. '.join(sentences[:2]).strip()
                        if brief and not brief.endswith('.'):
                            brief += '.'
                        accumulated_context += f"- {sec_title}: {brief}\n"

        return summaries

    def _is_summarizable_section(self, title: str) -> bool:
        """요약할 가치가 있는 섹션인지 판단합니다."""
        skip_patterns = [
            "abstract", "reference", "acknowledgment", "appendix",
            "table of contents", "contents", "bibliography"
        ]
        title_lower = title.lower()

        for pattern in skip_patterns:
            if pattern in title_lower:
                return False

        return True

    def _clean_thinking_tokens(self, text: str) -> str:
        """deepseek-r1 등의 <think>...</think> 토큰을 제거합니다."""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def _extract_section_content(self, markdown: str, section_title: str) -> str:
        """마크다운에서 특정 섹션의 내용을 추출합니다."""
        escaped_title = re.escape(section_title)
        pattern = r"#{1,6}\s+" + escaped_title + r"\s*\n(.*?)(?=\n#{1,6}\s+|\Z)"
        match = re.search(pattern, markdown, re.DOTALL)

        if match:
            return match.group(1).strip()

        return ""


def summarize_paper(
    structure: PaperStructure,
    provider: str = "ollama",
    model: str = "llama3.2",
    api_key: str | None = None,
    language: str = "korean",
) -> dict[str, str]:
    """논문을 요약하는 편의 함수"""
    config = SummaryConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        language=language,
    )
    summarizer = Summarizer(config)
    return summarizer.summarize_paper(structure)
