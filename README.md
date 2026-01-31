# Paper Summary Pipeline

논문 PDF를 Obsidian 호환 마크다운으로 자동 변환하는 파이프라인입니다.

## 주요 업데이트 (v2.0)

- **PDF 파싱 엔진**: marker-pdf 기반 (고품질 Figure/Table 추출)
- **Apple Silicon 지원**: MPS 가속 (M1/M2/M3 Mac)
- **병렬 처리**: 이미지 저장 + LLM 요약 동시 실행
- **Qwen3 Thinking Mode**: 고품질 한국어 요약
- **모델 캐싱**: 여러 PDF 처리 시 모델 재사용으로 성능 향상

## 기능

- **PDF 구조 추출**: marker-pdf를 사용하여 섹션, Figure, Table 구조 자동 파싱
- **이미지 추출**: Figure와 Table을 PNG 이미지로 저장 (논문 약어 접두사 포함)
- **LLM 요약**: 섹션별 한국어 요약 생성 (Ollama, OpenAI, Anthropic 지원)
- **Obsidian 마크다운**: 목차, frontmatter, 내부 링크가 포함된 마크다운 생성
- **폴더 감시**: 자동으로 새 PDF 파일 감지 및 처리

## 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e .
```

### Apple Silicon (M1/M2/M3) 사용자

MPS 가속이 자동 활성화됩니다. 환경변수로 제어 가능:

```bash
# MPS 사용 (기본값)
export TORCH_DEVICE=mps

# MPS 비활성화 (CPU 사용)
export TORCH_DEVICE=cpu
```

## 사용법

### 1. 단일 PDF 처리

```bash
python main.py process paper.pdf
```

### 2. 폴더 내 모든 PDF 처리

```bash
python main.py process-all
# 또는 특정 폴더 지정
python main.py process-all -f /path/to/pdfs
```

### 3. 폴더 감시 모드

```bash
# target-pdf 폴더에 PDF를 넣으면 자동 처리
python main.py watch

# 기존 파일도 함께 처리
python main.py watch --process-existing
```

### 4. 설정 파일 생성

```bash
python main.py init
```

## 설정

`config.yaml` 파일에서 설정을 변경할 수 있습니다:

```yaml
llm:
  provider: "ollama"  # ollama, openai, anthropic
  model: "qwen3:14b"  # 권장: Qwen3 Thinking Mode
  temperature: 0.6    # Thinking mode 권장값

paths:
  watch_dir: "target-pdf"
  output_dir: "output"

output:
  summary_language: "korean"
```

### LLM 제공자 설정

#### Ollama (로컬, 무료) - 권장

```yaml
llm:
  provider: "ollama"
  model: "qwen3:14b"      # Thinking mode 지원
  temperature: 0.6        # Thinking mode 권장값
  max_tokens: 2048
```

다른 Ollama 모델도 사용 가능:
- `deepseek-r1:32b` - 고품질 reasoning (32GB+ RAM 권장)
- `llama3.2` - 빠른 처리
- `mistral` - 균형잡힌 성능

#### OpenAI

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
```

#### Anthropic

```yaml
llm:
  provider: "anthropic"
  model: "claude-3-haiku-20240307"
  api_key: "${ANTHROPIC_API_KEY}"
```

## 성능 최적화

### 모델 사전 로딩

marker-pdf는 5개의 딥러닝 모델(~750MB)을 사용합니다:
- TextDetectionModel (텍스트 영역 검출)
- RecognitionModel (OCR)
- LayoutModel (레이아웃 분석)
- OrderingModel (읽기 순서)
- TableRecModel (테이블 인식)

**예상 처리 시간:**
- 첫 실행 (모델 로드): 30-60초
- 이후 실행 (모델 캐싱): 5-10초

`watch` 및 `process-all` 모드에서는 자동으로 모델을 미리 로드합니다.

### 병렬 처리

파이프라인은 다음을 병렬로 실행합니다:
- 이미지 저장 + LLM 요약 (동시 실행)
- 섹션별 요약 (ThreadPoolExecutor)

## 출력 구조

```
output/
└── 논문제목/
    ├── 논문제목.md      # Obsidian 마크다운
    └── assets/
        ├── CLIP_fig1.png
        ├── CLIP_fig2.png
        └── CLIP_table1.png
```

## 출력 마크다운 예시

```markdown
---
title: "Learning Transferable Visual Models"
aliases: ["CLIP"]
tags: [paper]
date: 2025-01-24
---

# Learning Transferable Visual Models

> **[Overview]**
> CLIP은 자연어 supervision을 통해 시각적 표현을 학습하는 방법을 제안합니다...

## 목차
- [[#Abstract]]
- [[#1. Introduction]]
- [[#2. Approach]]

## Abstract

> **[요약]**
> 본 논문에서는 이미지와 텍스트를 동시에 학습하는...

![[assets/CLIP_fig1.png]]
*Figure 1: CLIP 아키텍처*
```

## 의존성

주요 라이브러리:
- `marker-pdf`: PDF 파싱 및 마크다운 변환
- `surya-ocr`: OCR 및 레이아웃 분석 (MPS 지원 fork)
- `litellm`: 다중 LLM 제공자 통합
- `watchdog`: 폴더 감시

## 문제 해결

### MPS 관련 경고

```
TableRecEncoderDecoderModel is not compatible with mps
```

이 경고는 정상입니다. 테이블 인식 모델은 자동으로 CPU로 폴백되며, 다른 모델들은 MPS를 사용합니다.

### 모델 다운로드

첫 실행 시 HuggingFace에서 모델을 다운로드합니다. 캐시 경로:
- `~/.cache/huggingface/`
- `~/.cache/torch/`

## 라이선스

MIT
