# coding:utf-8
import functools
import time
import timeit

from improved_backoff._common import (_init_wait_gen, _maybe_call, _next_wait)


def _call_handlers(hdlrs, target, args, kwargs, tries, elapsed, **extra):
    details = {
        'target': target,
        'args': args,
        'kwargs': kwargs,
        'tries': tries,
        'elapsed': elapsed,
    }
    details.update(extra)
    for hdlr in hdlrs:
        hdlr(details)


def retry_predicate(target, wait_gen, predicate,
                    *,
                    max_tries, max_time, jitter,
                    on_success, on_backoff, on_giveup,
                    wait_gen_kwargs):

    @functools.wraps(target)
    def retry(*args, **kwargs):
        max_tries_value = _maybe_call(max_tries)
        max_time_value = _maybe_call(max_time)

        tries = 0
        start = timeit.default_timer()
        wait = _init_wait_gen(wait_gen, wait_gen_kwargs)
        while True:
            tries += 1
            ret = target(*args, **kwargs)
            elapsed = timeit.default_timer() - start
            details = {
                "target": target,
                "args": args,
                "kwargs": kwargs,
                "tries": tries,
                "elapsed": elapsed,
            }

            if predicate(ret):
                try:
                    seconds = _next_wait(wait, ret, jitter, elapsed,
                                         max_time_value)
                except StopIteration:
                    _call_handlers(on_giveup, **details)
                    break

                max_tries_exceeded = (tries == max_tries_value)
                max_time_exceeded = (max_time_value is not None) and (
                    elapsed >= max_time_value - seconds
                )

                if max_tries_exceeded or max_time_exceeded:
                    _call_handlers(on_giveup, **details, value=ret)
                    break

                _call_handlers(on_backoff, **details,
                               value=ret, wait=seconds)

                time.sleep(seconds)
                continue
            else:
                _call_handlers(on_success, **details, value=ret)
                break

        return ret

    return retry


def retry_exception(target, wait_gen, exception,
                    *,
                    max_tries, max_time, jitter, giveup,
                    on_success, on_backoff, on_giveup, raise_on_giveup,
                    wait_gen_kwargs):

    @functools.wraps(target)
    def retry(*args, **kwargs):
        max_tries_value = _maybe_call(max_tries)
        max_time_value = _maybe_call(max_time)

        tries = 0
        start = timeit.default_timer()
        wait = _init_wait_gen(wait_gen, wait_gen_kwargs)
        while True:
            tries += 1

            try:
                ret = target(*args, **kwargs)
            except exception as e:
                elapsed = timeit.default_timer() - start
                details = {
                    "target": target,
                    "args": args,
                    "kwargs": kwargs,
                    "tries": tries,
                    "elapsed": elapsed,
                }

                try:
                    seconds = _next_wait(wait, e, jitter, elapsed,
                                         max_time_value)
                except StopIteration:
                    _call_handlers(on_giveup, **details, exception=e)
                    raise e

                max_tries_exceeded = (tries == max_tries_value)
                max_time_exceeded = (max_time_value is not None) and (
                    elapsed >= max_time_value - seconds
                )

                if giveup(e) or max_tries_exceeded or max_time_exceeded:
                    _call_handlers(on_giveup, **details, exception=e)
                    if raise_on_giveup:
                        raise
                    return None

                _call_handlers(on_backoff, **details, wait=seconds,
                               exception=e)

                time.sleep(seconds)
            else:
                elapsed = timeit.default_timer() - start
                details = {
                    "target": target,
                    "args": args,
                    "kwargs": kwargs,
                    "tries": tries,
                    "elapsed": elapsed,
                }
                _call_handlers(on_success, **details)

                return ret
    return retry
