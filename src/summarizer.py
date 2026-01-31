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

SECTION_SUMMARY_PROMPT = """# TASK
Summarize the following paper section for a {language} researcher.

{context}
# SECTION INFO
- Section Title: "{title}"
- Content:
{content}

# CRITICAL RULES
1. **LANGUAGE:** Write **ONLY in {language}**. Technical terms in English allowed.
2. **FORMAT:**
   - Write in **prose (줄글)** style — natural flowing sentences, NOT bullet points.
   - **NO Markdown Headings** (`#`, `##`) are allowed.
   - Use **Bold text** for key concepts.
3. **ADAPTIVITY:**
   - Adapt your summary style to match the section's nature.
   - For example: "Introduction" → focus on motivation and problem statement;
     "Methods" → focus on proposed approach and technical details;
     "Experiments/Results" → focus on benchmarks, metrics, and comparisons;
     "Related Work" → focus on positioning against prior work;
     "Discussion/Conclusion" → focus on implications and limitations.
   - If specific numbers/metrics are present, MUST include them.
4. **LENGTH:**
   - Short section (< 500 words): 2-4 sentences.
   - Long section (≥ 500 words): 5-8 sentences max.
5. **CONTEXT:** Ensure continuity with the previous section summary.

# OUTPUT EXAMPLE
이 섹션에서는 기존 Attention 메커니즘의 $O(N^2)$ 연산 복잡도 문제를 지적하고, 이를 $O(N)$으로 줄이는 **Linear Attention**을 제안한다. 핵심 아이디어는 커널 함수 $\\phi(\\cdot)$를 활용하여 행렬 곱 연산 순서를 변경하는 것이며, 이를 통해 기존 대비 학습 속도가 2.5배 향상되었다."""

MARKDOWN_SUMMARY_PROMPT = """# TASK
Summarize each main section of the provided markdown paper.

# INPUT TEXT
{markdown}

# REQUIREMENTS
1. **Language:** {language} Only.
2. **Length:** 2-3 sentences per section.
3. **Format:** Output using the specific structure below.

# OUTPUT FORMAT
## [Section Title]
[Summary content in {language}]

## [Next Section Title]
[Summary content in {language}]"""

PAPER_OVERVIEW_PROMPT = """# TASK
Synthesize a comprehensive overview of the entire paper based on the section summaries.

# PAPER INFO
- Title: {title}
- Section Summaries:
{section_summaries}

# CRITICAL RULES
1. **LANGUAGE:** Write the summary **ONLY in {language}**.
   - Technical terms in English allowed.
2. **FORMAT:**
   - Use bullet points (`-`) strictly.
   - **NO Markdown Headings** (`#`, `##`) are allowed.
3. **STRUCTURE (Strict Order, 5-10 bullets total):**
   (1) Research Motivation/Background (1 bullet)
   (2) Core Methodology (1-2 bullets)
   (3) Key Contributions (1-2 bullets)
   (4) Key Experimental Results — MUST include numbers/benchmarks (1 bullet)
   (5) Significance or Limitation (1 bullet)
4. **MATHEMATICS:** Keep raw LaTeX format (e.g., $\\alpha$, $L_{{total}}$).

# OUTPUT EXAMPLE (Follow this style)
- **연구 배경**: LLM의 환각(Hallucination) 문제를 해결하기 위한 데이터 정제 필요성 대두
- **핵심 방법론**: 
  - 강화학습(RLHF) 단계에서 새로운 보상 함수 $R(x)$ 제안
  - PPO 알고리즘을 개선한 'PPO-Clip-v2' 도입
- **주요 기여**: 기존 방법론 대비 연산 비용을 30% 절감하면서 성능 유지
- **실험 결과**: TruthfulQA 벤치마크에서 SOTA 달성 (Accuracy: 85.4% vs 82.1%)
- **한계점**: 영어 데이터셋에만 검증되었으며 다국어 성능은 미지수"""

METADATA_EXTRACT_PROMPT = """# TASK
Extract metadata from the beginning of the paper text.

# INPUT TEXT
{text}

# REQUIREMENTS
1. Output **ONLY valid JSON** format.
2. No conversational text or markdown blocks (```json).
3. Use `null` if the information is missing.

# JSON SCHEMA
{{
    "authors": ["Author 1", "Author 2", ...], 
    "journal": "Journal/Conference Name", 
    "year": "YYYY"
}}"""




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


    def get_system_prompt(self) -> str:
        """시스템 프롬프트를 반환합니다."""
        return f"""# ROLE
You are a senior researcher and expert in summarizing academic papers in ML/AI/CS (Deep Learning, RL, CV, NLP).

# TASK
Analyze the given paper content and summarize it effectively for a Korean researcher.

# CRITICAL RULES (MUST FOLLOW)
1. **OUTPUT LANGUAGE:** You must write the summary **ONLY in {self.config.language} (Korean).**
   - DO NOT write the summary in English.
   - Exception: Keep technical terms in English (e.g., Transformer, PPO, Zero-shot, Backbone).
2. **FORMAT:**
   - Write in **prose (줄글)** style — natural flowing paragraph, NOT bullet points.
   - **NO Markdown Headings** (`#`, `##`) are allowed. They conflict with the user's document structure.
   - Use **Bold text** for key concepts.
3. **CONTENT:**
   - Capture the core contribution, methodology, and quantitative results.
   - Be concise and professional. No introductory filler words.
4. **MATHEMATICS:** Keep raw LaTeX format (e.g., $\\alpha$, $L_{{total}}$).

# REMINDER
- Output **MUST** be in Korean.
- Do not use `#` headings.
- Write in prose, not bullet points."""

    def _call_llm(self, prompt: str) -> str:
        """LLM을 호출하여 응답을 받습니다."""
        model_name = self._get_model_name()
        system_prompt = self.get_system_prompt()

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
                if section_content:
                    sections_to_summarize.append((section, section_content))

            # 배치별 병렬 처리
            def summarize_single_section(args, prior_context=""):
                section, content = args
                context_block = ""
                if prior_context:
                    context_block = f"\n# Summary of previous section\n{prior_context}\n"
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
