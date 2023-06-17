import concurrent.futures
import itertools
import os
from typing import Callable, Iterable


def execute_multiprocess(func: Callable, inputs: Iterable, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count()

    executor_cls = concurrent.futures.ProcessPoolExecutor
    yield from _execute_parallel(func, inputs, executor_cls, n_workers)


def execute_multithread(func: Callable, inputs: Iterable, n_workers):
    executor_cls = concurrent.futures.ThreadPoolExecutor
    yield from _execute_parallel(func, inputs, executor_cls, n_workers)


def _execute_parallel(func: Callable, inputs: Iterable, executor_cls, n_workers: int):
    with executor_cls(max_workers=n_workers) as executor:
        futures = {
            executor.submit(func, **task)
            for task in itertools.islice(inputs, n_workers)
        }

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                yield future.result()

            for task in itertools.islice(inputs, len(done)):
                futures.add(executor.submit(func, **task))
