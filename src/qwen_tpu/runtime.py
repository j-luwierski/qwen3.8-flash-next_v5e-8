"""Cooperative wall-clock deadlines and portable, atomic resume files.

A deadline is checked at safe boundaries, never by killing a JAX computation.
The reserved margin must cover the longest noninterruptible operation + saving.
"""

import os
import math
import signal
import threading
import time
from pathlib import Path

from .io import read_json, sha256, write_json


class SessionPaused(RuntimeError):
    def __init__(self, stage, reason='time_budget'):
        self.stage, self.reason = stage, reason
        super().__init__(f'Paused {stage}: {reason}; saved work can be resumed')


class RunControl:
    def __init__(self, max_run_seconds=None, save_margin_seconds=300,
                 checkpoint_interval_seconds=300, *, clock=None):
        if max_run_seconds is not None and (not math.isfinite(max_run_seconds) or max_run_seconds < 0):
            raise ValueError('max_run_seconds must be nonnegative or null')
        if not math.isfinite(save_margin_seconds) or not math.isfinite(checkpoint_interval_seconds) or save_margin_seconds < 0 or checkpoint_interval_seconds <= 0:
            raise ValueError('Nonnegative save margin and positive checkpoint interval required')
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.limit, self.margin, self.interval = max_run_seconds, save_margin_seconds, checkpoint_interval_seconds
        self.last_saved = self.started
        self.event = threading.Event()
        self.reason = 'time_budget'
        self.handlers = {}

    def elapsed(self):
        return self.clock() - self.started

    def request_stop(self, reason='requested'):
        self.reason = reason
        self.event.set()

    def should_stop(self):
        return self.event.is_set() or (self.limit is not None and self.elapsed() >= max(0, self.limit-self.margin))

    def check(self, stage):
        if self.should_stop():
            raise SessionPaused(stage, self.reason)

    def due(self, last_saved=None):
        return self.clock() - (self.last_saved if last_saved is None else last_saved) >= self.interval

    def saved(self):
        self.last_saved = self.clock()

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda number, frame: self.request_stop(f'signal_{number}'))
        return self

    def __exit__(self, *_args):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)


def check(control, stage):
    if control:
        control.check(stage)


def resumable_copy(source, destination, expected_sha256, control=None, chunk_bytes=8*1024**2):
    """A .copying file is an unfinished prefix; publish only after full checksum."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f'Copied file checksum mismatch: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.copying')
    size = partial.stat().st_size if partial.exists() else 0
    if size > source.stat().st_size:
        raise ValueError('Partial copy is longer than its source')
    with source.open('rb') as src, partial.open('ab') as dst:
        src.seek(size)
        try:
            while True:
                check(control, 'copy')
                block = src.read(chunk_bytes)
                if not block:
                    break
                dst.write(block)
        finally:
            dst.flush()
            os.fsync(dst.fileno())
    if sha256(partial) != expected_sha256:
        raise ValueError(f'Source/partial copy checksum mismatch: {source}')
    os.replace(partial, destination)


def validation_loop(progress, count, evaluate, control, save):
    """Persist a validation cursor without changing the training step counter."""
    while progress['next_index'] < count:
        check(control, 'validation')
        value = float(evaluate(progress['next_index']))
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite validation loss')
        progress['losses'].append(value)
        progress['next_index'] += 1
        if control and control.due():
            save()
    return sum(progress['losses']) / len(progress['losses'])
