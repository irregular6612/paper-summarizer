#!/usr/bin/env python3
"""Paper Summary Pipeline - 논문 PDF를 Obsidian 마크다운으로 변환"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config, save_default_config
from src.pipeline import PaperPipeline
from src.watcher import FolderWatcher
from src.extractor import warmup_extractor


def setup_logging(verbose: bool = False) -> None:
    """로깅을 설정합니다."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def warmup_models() -> None:
    """
    PDF 처리 모델을 미리 로드합니다.
    
    첫 PDF 처리 전 호출하면 대기 시간을 줄일 수 있습니다.
    - 첫 실행: 30-60초 (모델 로드)
    - 이후 실행: 5-10초 (PDF 파싱만)
    """
    logging.info("Warming up PDF processing models...")
    warmup_extractor()


def cmd_process(args: argparse.Namespace) -> int:
    """단일 PDF 파일을 처리합니다."""
    config = load_config(args.config)
    pipeline = PaperPipeline(config)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logging.error(f"File not found: {pdf_path}")
        return 1

    try:
        output_path = pipeline.process(pdf_path)
        logging.info(f"Successfully created: {output_path}")
        return 0
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        return 1


def cmd_watch(args: argparse.Namespace) -> int:
    """폴더를 감시하고 PDF를 자동 처리합니다."""
    config = load_config(args.config)
    
    # 감시 시작 전 모델 미리 로드
    warmup_models()
    
    pipeline = PaperPipeline(config)

    watch_dir = Path(config.paths.watch_dir)
    processed_dir = Path(config.paths.processed_dir) if config.paths.processed_dir else None

    def process_callback(pdf_path: Path) -> None:
        try:
            pipeline.process(pdf_path)
        except Exception as e:
            logging.error(f"Failed to process {pdf_path.name}: {e}")

    watcher = FolderWatcher(
        watch_dir=watch_dir,
        callback=process_callback,
        processed_dir=processed_dir,
    )

    # 기존 파일 처리
    if args.process_existing:
        logging.info("Processing existing PDF files...")
        watcher.process_existing()

    logging.info(f"Watching folder: {watch_dir}")
    logging.info("Press Ctrl+C to stop")

    try:
        watcher.start(blocking=True)
    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
        watcher.stop()

    return 0


def cmd_process_all(args: argparse.Namespace) -> int:
    """폴더 내 모든 PDF 파일을 처리합니다."""
    config = load_config(args.config)
    
    # 처리 시작 전 모델 미리 로드
    warmup_models()
    
    pipeline = PaperPipeline(config)

    watch_dir = Path(args.folder) if args.folder else Path(config.paths.watch_dir)
    
    if not watch_dir.exists():
        logging.error(f"Folder not found: {watch_dir}")
        return 1

    pdf_files = list(watch_dir.glob("*.pdf"))
    if not pdf_files:
        logging.info(f"No PDF files found in {watch_dir}")
        return 0

    logging.info(f"Found {len(pdf_files)} PDF files to process")
    
    success_count = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing [{i}/{len(pdf_files)}]: {pdf_path.name}")
        logging.info(f"{'='*60}")
        
        try:
            output_path = pipeline.process(pdf_path)
            logging.info(f"Successfully created: {output_path}")
            success_count += 1
        except Exception as e:
            logging.error(f"Failed to process {pdf_path.name}: {e}")
            if args.verbose:
                logging.exception("Full traceback:")

    logging.info(f"\n{'='*60}")
    logging.info(f"Completed: {success_count}/{len(pdf_files)} files processed successfully")
    return 0 if success_count == len(pdf_files) else 1


def cmd_init(args: argparse.Namespace) -> int:
    """기본 설정 파일을 생성합니다."""
    config_path = Path(args.output)

    if config_path.exists() and not args.force:
        logging.error(f"Config file already exists: {config_path}")
        logging.error("Use --force to overwrite")
        return 1

    save_default_config(config_path)
    logging.info(f"Created config file: {config_path}")
    return 0


def main() -> int:
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        description="Paper Summary Pipeline - 논문 PDF를 Obsidian 마크다운으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 PDF 처리
  python main.py process paper.pdf

  # 폴더 감시 모드
  python main.py watch

  # 기본 설정 파일 생성
  python main.py init
        """,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="설정 파일 경로 (기본: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # process 명령어
    process_parser = subparsers.add_parser(
        "process",
        help="단일 PDF 파일 처리",
    )
    process_parser.add_argument(
        "pdf",
        help="처리할 PDF 파일 경로",
    )

    # watch 명령어
    watch_parser = subparsers.add_parser(
        "watch",
        help="폴더 감시 모드",
    )
    watch_parser.add_argument(
        "--process-existing",
        action="store_true",
        help="시작 시 기존 PDF 파일도 처리",
    )

    # process-all 명령어
    process_all_parser = subparsers.add_parser(
        "process-all",
        help="폴더 내 모든 PDF 파일 처리",
    )
    process_all_parser.add_argument(
        "-f", "--folder",
        help="처리할 PDF 폴더 (기본: config의 watch_dir)",
    )

    # init 명령어
    init_parser = subparsers.add_parser(
        "init",
        help="기본 설정 파일 생성",
    )
    init_parser.add_argument(
        "-o", "--output",
        default="config.yaml",
        help="설정 파일 저장 경로",
    )
    init_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="기존 파일 덮어쓰기",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "process": cmd_process,
        "process-all": cmd_process_all,
        "watch": cmd_watch,
        "init": cmd_init,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
