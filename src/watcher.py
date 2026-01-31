"""폴더 감시 모듈 - watchdog 기반 자동 처리"""

import logging
import time
from pathlib import Path
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)


class PDFHandler(FileSystemEventHandler):
    """PDF 파일 생성을 감지하는 핸들러"""

    def __init__(
        self,
        callback: Callable[[Path], None],
        processed_dir: Path | None = None,
        debounce_seconds: float = 2.0,
    ):
        """
        Args:
            callback: PDF 파일이 감지되면 호출할 콜백 함수
            processed_dir: 처리 완료된 파일을 이동할 디렉토리 (None이면 이동 안함)
            debounce_seconds: 파일 쓰기 완료 대기 시간
        """
        super().__init__()
        self.callback = callback
        self.processed_dir = processed_dir
        self.debounce_seconds = debounce_seconds
        self._pending_files: dict[str, float] = {}
        self._processed_files: set[str] = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        """파일 생성 이벤트 처리"""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # PDF 파일만 처리
        if path.suffix.lower() != ".pdf":
            return

        # 이미 처리된 파일 스킵
        if str(path) in self._processed_files:
            return

        # 디바운스: 파일 쓰기 완료 대기
        self._pending_files[str(path)] = time.time()

    def on_modified(self, event) -> None:
        """파일 수정 이벤트 처리 (대용량 파일 복사 시)"""
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() != ".pdf":
            return

        # pending 상태면 시간 갱신
        if str(path) in self._pending_files:
            self._pending_files[str(path)] = time.time()

    def process_pending(self) -> None:
        """대기 중인 파일들을 처리합니다."""
        current_time = time.time()
        to_process = []

        for path_str, timestamp in list(self._pending_files.items()):
            # 디바운스 시간이 지났으면 처리
            if current_time - timestamp >= self.debounce_seconds:
                to_process.append(path_str)

        for path_str in to_process:
            del self._pending_files[path_str]
            self._processed_files.add(path_str)

            path = Path(path_str)
            if path.exists():
                logger.info(f"Processing: {path.name}")
                try:
                    self.callback(path)
                    logger.info(f"Completed: {path.name}")

                    # 처리 완료 후 이동
                    if self.processed_dir:
                        self.processed_dir.mkdir(parents=True, exist_ok=True)
                        new_path = self.processed_dir / path.name
                        path.rename(new_path)
                        logger.info(f"Moved to: {new_path}")

                except Exception as e:
                    logger.error(f"Error processing {path.name}: {e}")


class FolderWatcher:
    """폴더를 감시하고 PDF 파일을 자동 처리하는 클래스"""

    def __init__(
        self,
        watch_dir: Path,
        callback: Callable[[Path], None],
        processed_dir: Path | None = None,
        poll_interval: float = 1.0,
    ):
        """
        Args:
            watch_dir: 감시할 디렉토리
            callback: PDF 감지 시 호출할 콜백
            processed_dir: 처리 완료 파일 이동 디렉토리
            poll_interval: 폴링 간격 (초)
        """
        self.watch_dir = watch_dir
        self.callback = callback
        self.processed_dir = processed_dir
        self.poll_interval = poll_interval
        self.observer: Observer | None = None
        self.handler: PDFHandler | None = None

    def start(self, blocking: bool = True) -> None:
        """
        폴더 감시를 시작합니다.

        Args:
            blocking: True면 Ctrl+C까지 블로킹, False면 백그라운드 실행
        """
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        self.handler = PDFHandler(
            callback=self.callback,
            processed_dir=self.processed_dir,
        )

        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()

        logger.info(f"Watching: {self.watch_dir}")

        if blocking:
            try:
                while True:
                    time.sleep(self.poll_interval)
                    self.handler.process_pending()
            except KeyboardInterrupt:
                self.stop()
        else:
            # 백그라운드 실행 시 별도 스레드에서 pending 처리 필요
            import threading

            def poll_loop():
                while self.observer and self.observer.is_alive():
                    time.sleep(self.poll_interval)
                    if self.handler:
                        self.handler.process_pending()

            self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
            self._poll_thread.start()

    def stop(self) -> None:
        """폴더 감시를 중지합니다."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Watcher stopped")

    def process_existing(self) -> None:
        """이미 존재하는 PDF 파일들을 처리합니다."""
        for pdf_path in self.watch_dir.glob("*.pdf"):
            logger.info(f"Processing existing: {pdf_path.name}")
            try:
                self.callback(pdf_path)
                logger.info(f"Completed: {pdf_path.name}")

                if self.processed_dir:
                    self.processed_dir.mkdir(parents=True, exist_ok=True)
                    new_path = self.processed_dir / pdf_path.name
                    pdf_path.rename(new_path)
                    logger.info(f"Moved to: {new_path}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
