# coding:utf-8
import datetime
import itertools
import logging
import random
import re
import sys
import threading
import unittest.mock

import pytest

import improved_backoff as backoff
from tests.common import _save_target


def test_on_predicate(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    @backoff.on_predicate(backoff.expo)
    def return_true(log, n):
        val = (len(log) == n - 1)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 3)
    assert ret is True
    assert 3 == len(log)


def test_on_predicate_max_tries(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    @backoff.on_predicate(backoff.expo, jitter=None, max_tries=3)
    def return_true(log, n):
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 10)
    assert ret is False
    assert 3 == len(log)


def test_on_exception(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    @backoff.on_exception(backoff.expo, KeyError)
    def keyerror_then_true(log, n):
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    assert keyerror_then_true(log, 3) is True
    assert 3 == len(log)


def test_on_exception_tuple(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    @backoff.on_exception(backoff.expo, (KeyError, ValueError))
    def keyerror_valueerror_then_true(log):
        if len(log) == 2:
            return True
        if len(log) == 0:
            e = KeyError()
        if len(log) == 1:
            e = ValueError()
        log.append(e)
        raise e

    log = []
    assert keyerror_valueerror_then_true(log) is True
    assert 2 == len(log)
    assert isinstance(log[0], KeyError)
    assert isinstance(log[1], ValueError)


def test_on_exception_max_tries(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    @backoff.on_exception(backoff.expo, KeyError, jitter=None, max_tries=3)
    def keyerror_then_true(log, n, foo=None):
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10, foo="bar")

    assert 3 == len(log)


def test_on_exception_max_tries_callable(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    @backoff.on_exception(backoff.expo, KeyError, jitter=None,
                          max_tries=lambda: 3)
    def keyerror_then_true(log, n, foo=None):
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10, foo="bar")

    assert 3 == len(log)


def test_on_exception_constant_iterable(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    backoffs = []
    giveups = []
    successes = []

    @backoff.on_exception(
        backoff.constant,
        KeyError,
        interval=(1, 2, 3),
        on_backoff=backoffs.append,
        on_giveup=giveups.append,
        on_success=successes.append,
    )
    def endless_exceptions():
        raise KeyError('foo')

    with pytest.raises(KeyError):
        endless_exceptions()

    assert len(backoffs) == 3
    assert len(giveups) == 1
    assert len(successes) == 0


def test_on_exception_success_random_jitter(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    backoffs, giveups, successes = [], [], []

    @backoff.on_exception(backoff.expo,
                          Exception,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          jitter=backoff.random_jitter,
                          factor=0.5)
    @_save_target
    def succeeder(*args, **kwargs):
        # succeed after we've backed off twice
        if len(backoffs) < 2:
            raise ValueError("catch me")

    succeeder(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice before succeeding
    assert len(successes) == 1
    assert len(backoffs) == 2
    assert len(giveups) == 0

    for i in range(2):
        details = backoffs[i]
        assert details['wait'] >= 0.5 * 2 ** i


def test_on_exception_success_full_jitter(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    backoffs, giveups, successes = [], [], []

    @backoff.on_exception(backoff.expo,
                          Exception,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          jitter=backoff.full_jitter,
                          factor=0.5)
    @_save_target
    def succeeder(*args, **kwargs):
        # succeed after we've backed off twice
        if len(backoffs) < 2:
            raise ValueError("catch me")

    succeeder(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice before succeeding
    assert len(successes) == 1
    assert len(backoffs) == 2
    assert len(giveups) == 0

    for i in range(2):
        details = backoffs[i]
        assert details['wait'] <= 0.5 * 2 ** i


def test_on_exception_success():
    backoffs, giveups, successes = [], [], []

    @backoff.on_exception(backoff.constant,
                          Exception,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          jitter=None,
                          interval=0)
    @_save_target
    def succeeder(*args, **kwargs):
        # succeed after we've backed off twice
        if len(backoffs) < 2:
            raise ValueError("catch me")

    succeeder(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice before succeeding
    assert len(successes) == 1
    assert len(backoffs) == 2
    assert len(giveups) == 0

    for i in range(2):
        details = backoffs[i]
        elapsed = details.pop('elapsed')
        exception = details.pop('exception')
        assert isinstance(elapsed, float)
        assert isinstance(exception, ValueError)
        assert details == {'args': (1, 2, 3),
                           'kwargs': {'foo': 1, 'bar': 2},
                           'target': succeeder._target,
                           'tries': i + 1,
                           'wait': 0}

    details = successes[0]
    elapsed = details.pop('elapsed')
    assert isinstance(elapsed, float)
    assert details == {'args': (1, 2, 3),
                       'kwargs': {'foo': 1, 'bar': 2},
                       'target': succeeder._target,
                       'tries': 3}


@pytest.mark.parametrize('raise_on_giveup', [True, False])
def test_on_exception_giveup(raise_on_giveup):
    backoffs, giveups, successes = [], [], []

    @backoff.on_exception(backoff.constant,
                          ValueError,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          max_tries=3,
                          jitter=None,
                          raise_on_giveup=raise_on_giveup,
                          interval=0)
    @_save_target
    def exceptor(*args, **kwargs):
        raise ValueError("catch me")

    if raise_on_giveup:
        with pytest.raises(ValueError):
            exceptor(1, 2, 3, foo=1, bar=2)
    else:
        exceptor(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice and giving up once
    assert len(successes) == 0
    assert len(backoffs) == 2
    assert len(giveups) == 1

    details = giveups[0]
    elapsed = details.pop('elapsed')
    exception = details.pop('exception')
    assert isinstance(elapsed, float)
    assert isinstance(exception, ValueError)
    assert details == {'args': (1, 2, 3),
                       'kwargs': {'foo': 1, 'bar': 2},
                       'target': exceptor._target,
                       'tries': 3}


def test_on_exception_giveup_predicate(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    def on_baz(e):
        return str(e) == "baz"

    vals = ["baz", "bar", "foo"]

    @backoff.on_exception(backoff.constant,
                          ValueError,
                          giveup=on_baz)
    def foo_bar_baz():
        raise ValueError(vals.pop())

    with pytest.raises(ValueError):
        foo_bar_baz()

    assert not vals


def test_on_predicate_success():
    backoffs, giveups, successes = [], [], []

    @backoff.on_predicate(backoff.constant,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          jitter=None,
                          interval=0)
    @_save_target
    def success(*args, **kwargs):
        # succeed after we've backed off twice
        return len(backoffs) == 2

    success(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice before succeeding
    assert len(successes) == 1
    assert len(backoffs) == 2
    assert len(giveups) == 0

    for i in range(2):
        details = backoffs[i]

        elapsed = details.pop('elapsed')
        assert isinstance(elapsed, float)
        assert details == {'args': (1, 2, 3),
                           'kwargs': {'foo': 1, 'bar': 2},
                           'target': success._target,
                           'tries': i + 1,
                           'value': False,
                           'wait': 0}

    details = successes[0]
    elapsed = details.pop('elapsed')
    assert isinstance(elapsed, float)
    assert details == {'args': (1, 2, 3),
                       'kwargs': {'foo': 1, 'bar': 2},
                       'target': success._target,
                       'tries': 3,
                       'value': True}


def test_on_predicate_giveup():
    backoffs, giveups, successes = [], [], []

    @backoff.on_predicate(backoff.constant,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          max_tries=3,
                          jitter=None,
                          interval=0)
    @_save_target
    def emptiness(*args, **kwargs):
        pass

    emptiness(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice and giving up once
    assert len(successes) == 0
    assert len(backoffs) == 2
    assert len(giveups) == 1

    details = giveups[0]
    elapsed = details.pop('elapsed')
    assert isinstance(elapsed, float)
    assert details == {'args': (1, 2, 3),
                       'kwargs': {'foo': 1, 'bar': 2},
                       'target': emptiness._target,
                       'tries': 3,
                       'value': None}


def test_on_predicate_iterable_handlers():
    class Logger:
        def __init__(self):
            self.backoffs = []
            self.giveups = []
            self.successes = []

    loggers = [Logger() for _ in range(3)]

    @backoff.on_predicate(backoff.constant,
                          on_backoff=(log.backoffs.append for log in loggers),
                          on_giveup=(log.giveups.append for log in loggers),
                          on_success=(log.successes.append for log in loggers),
                          max_tries=3,
                          jitter=None,
                          interval=0)
    @_save_target
    def emptiness(*args, **kwargs):
        pass

    emptiness(1, 2, 3, foo=1, bar=2)

    for logger in loggers:

        assert len(logger.successes) == 0
        assert len(logger.backoffs) == 2
        assert len(logger.giveups) == 1

        details = dict(logger.giveups[0])
        print(details)
        elapsed = details.pop('elapsed')
        assert isinstance(elapsed, float)
        assert details == {'args': (1, 2, 3),
                           'kwargs': {'foo': 1, 'bar': 2},
                           'target': emptiness._target,
                           'tries': 3,
                           'value': None}


# To maintain backward compatibility,
# on_predicate should support 0-argument jitter function.
def test_on_exception_success_0_arg_jitter(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)
    monkeypatch.setattr('random.random', lambda: 0)

    backoffs, giveups, successes = [], [], []

    @backoff.on_exception(backoff.constant,
                          Exception,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          jitter=random.random,
                          interval=0)
    @_save_target
    def succeeder(*args, **kwargs):
        # succeed after we've backed off twice
        if len(backoffs) < 2:
            raise ValueError("catch me")

    with pytest.deprecated_call():
        succeeder(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice before succeeding
    assert len(successes) == 1
    assert len(backoffs) == 2
    assert len(giveups) == 0

    for i in range(2):
        details = backoffs[i]
        elapsed = details.pop('elapsed')
        exception = details.pop('exception')
        assert isinstance(elapsed, float)
        assert isinstance(exception, ValueError)
        assert details == {'args': (1, 2, 3),
                           'kwargs': {'foo': 1, 'bar': 2},
                           'target': succeeder._target,
                           'tries': i + 1,
                           'wait': 0}

    details = successes[0]
    elapsed = details.pop('elapsed')
    assert isinstance(elapsed, float)
    assert details == {'args': (1, 2, 3),
                       'kwargs': {'foo': 1, 'bar': 2},
                       'target': succeeder._target,
                       'tries': 3}


# To maintain backward compatibility,
# on_predicate should support 0-argument jitter function.
def test_on_predicate_success_0_arg_jitter(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)
    monkeypatch.setattr('random.random', lambda: 0)

    backoffs, giveups, successes = [], [], []

    @backoff.on_predicate(backoff.constant,
                          on_success=successes.append,
                          on_backoff=backoffs.append,
                          on_giveup=giveups.append,
                          jitter=random.random,
                          interval=0)
    @_save_target
    def success(*args, **kwargs):
        # succeed after we've backed off twice
        return len(backoffs) == 2

    with pytest.deprecated_call():
        success(1, 2, 3, foo=1, bar=2)

    # we try 3 times, backing off twice before succeeding
    assert len(successes) == 1
    assert len(backoffs) == 2
    assert len(giveups) == 0

    for i in range(2):
        details = backoffs[i]
        print(details)
        elapsed = details.pop('elapsed')
        assert isinstance(elapsed, float)
        assert details == {'args': (1, 2, 3),
                           'kwargs': {'foo': 1, 'bar': 2},
                           'target': success._target,
                           'tries': i + 1,
                           'value': False,
                           'wait': 0}

    details = successes[0]
    elapsed = details.pop('elapsed')
    assert isinstance(elapsed, float)
    assert details == {'args': (1, 2, 3),
                       'kwargs': {'foo': 1, 'bar': 2},
                       'target': success._target,
                       'tries': 3,
                       'value': True}


def test_on_exception_callable_max_tries(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    log = []

    @backoff.on_exception(backoff.constant, ValueError, max_tries=lambda: 3)
    def exceptor():
        log.append(True)
        raise ValueError()

    with pytest.raises(ValueError):
        exceptor()

    assert len(log) == 3


def test_on_exception_callable_max_tries_reads_every_time(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    lookups = []

    def lookup_max_tries():
        lookups.append(True)
        return 3

    @backoff.on_exception(backoff.constant,
                          ValueError,
                          max_tries=lookup_max_tries)
    def exceptor():
        raise ValueError()

    with pytest.raises(ValueError):
        exceptor()

    with pytest.raises(ValueError):
        exceptor()

    assert len(lookups) == 2


def test_on_exception_callable_gen_kwargs():

    def lookup_foo():
        return "foo"

    def wait_gen(foo=None, bar=None):
        assert foo == "foo"
        assert bar == "bar"

        while True:
            yield 0

    @backoff.on_exception(wait_gen,
                          ValueError,
                          max_tries=2,
                          foo=lookup_foo,
                          bar="bar")
    def exceptor():
        raise ValueError("aah")

    with pytest.raises(ValueError):
        exceptor()


def test_on_predicate_in_thread(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    result = []

    def check():
        try:
            @backoff.on_predicate(backoff.expo)
            def return_true(log, n):
                val = (len(log) == n - 1)
                log.append(val)
                return val

            log = []
            ret = return_true(log, 3)
            assert ret is True
            assert 3 == len(log)

        except Exception as ex:
            result.append(ex)
        else:
            result.append('success')

    t = threading.Thread(target=check)
    t.start()
    t.join()

    assert len(result) == 1
    assert result[0] == 'success'


def test_on_predicate_constant_iterable(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    waits = [1, 2, 3, 6, 9]
    backoffs = []
    giveups = []
    successes = []

    @backoff.on_predicate(
        backoff.constant,
        interval=waits,
        on_backoff=backoffs.append,
        on_giveup=giveups.append,
        on_success=successes.append,
        jitter=None,
    )
    def falsey():
        return False

    assert not falsey()

    assert len(backoffs) == len(waits)
    for i, wait in enumerate(waits):
        assert backoffs[i]['wait'] == wait

    assert len(giveups) == 1
    assert len(successes) == 0


def test_on_exception_in_thread(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda x: None)

    result = []

    def check():
        try:
            @backoff.on_exception(backoff.expo, KeyError)
            def keyerror_then_true(log, n):
                if len(log) == n:
                    return True
                e = KeyError()
                log.append(e)
                raise e

            log = []
            assert keyerror_then_true(log, 3) is True
            assert 3 == len(log)

        except Exception as ex:
            result.append(ex)
        else:
            result.append('success')

    t = threading.Thread(target=check)
    t.start()
    t.join()

    assert len(result) == 1
    assert result[0] == 'success'


def test_on_exception_logger_default(monkeypatch, caplog):
    monkeypatch.setattr('time.sleep', lambda x: None)

    logger = logging.getLogger('backoff')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    @backoff.on_exception(backoff.expo, KeyError, max_tries=3)
    def key_error():
        raise KeyError()

    with caplog.at_level(logging.INFO):
        with pytest.raises(KeyError):
            key_error()

    assert len(caplog.records) == 3  # 2 backoffs and 1 giveup
    for record in caplog.records:
        assert record.name == 'backoff'


def test_on_exception_logger_none(monkeypatch, caplog):
    monkeypatch.setattr('time.sleep', lambda x: None)

    logger = logging.getLogger('backoff')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    @backoff.on_exception(backoff.expo, KeyError, max_tries=3, logger=None)
    def key_error():
        raise KeyError()

    with caplog.at_level(logging.INFO):
        with pytest.raises(KeyError):
            key_error()

    assert not caplog.records


def test_on_exception_logger_user(monkeypatch, caplog):
    monkeypatch.setattr('time.sleep', lambda x: None)

    logger = logging.getLogger('my-logger')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    @backoff.on_exception(backoff.expo, KeyError, max_tries=3, logger=logger)
    def key_error():
        raise KeyError()

    with caplog.at_level(logging.INFO):
        with pytest.raises(KeyError):
            key_error()

    assert len(caplog.records) == 3  # 2 backoffs and 1 giveup
    for record in caplog.records:
        assert record.name == 'my-logger'


def test_on_exception_logger_user_str(monkeypatch, caplog):
    monkeypatch.setattr('time.sleep', lambda x: None)

    logger = logging.getLogger('my-logger')
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    @backoff.on_exception(backoff.expo, KeyError, max_tries=3,
                          logger='my-logger')
    def key_error():
        raise KeyError()

    with caplog.at_level(logging.INFO):
        with pytest.raises(KeyError):
            key_error()

    assert len(caplog.records) == 3  # 2 backoffs and 1 giveup
    for record in caplog.records:
        assert record.name == 'my-logger'


def _on_exception_factory(
    backoff_log_level, giveup_log_level, max_tries
):
    @backoff.on_exception(
        backoff.expo,
        ValueError,
        max_tries=max_tries,
        backoff_log_level=backoff_log_level,
        giveup_log_level=giveup_log_level,
    )
    def value_error():
        raise ValueError

    def func():
        with pytest.raises(ValueError):
            value_error()

    return func


def _on_predicate_factory(
    backoff_log_level, giveup_log_level, max_tries
):
    @backoff.on_predicate(
        backoff.expo,
        max_tries=max_tries,
        backoff_log_level=backoff_log_level,
        giveup_log_level=giveup_log_level,
    )
    def func():
        return False

    return func


@pytest.mark.parametrize(
    ("func_factory", "backoff_log_level", "giveup_log_level"),
    (
        (factory, backoff_log_level, giveup_log_level)
        for backoff_log_level, giveup_log_level in itertools.product(
            (
                logging.DEBUG,
                logging.INFO,
                logging.WARNING,
                logging.ERROR,
                logging.CRITICAL,
            ),
            repeat=2,
        )
        for factory in (_on_predicate_factory, _on_exception_factory)
    )
)
def test_event_log_levels(
    caplog, func_factory, backoff_log_level, giveup_log_level
):
    max_tries = 3
    func = func_factory(backoff_log_level, giveup_log_level, max_tries)

    with unittest.mock.patch('time.sleep', return_value=None):
        with caplog.at_level(
            min(backoff_log_level, giveup_log_level), logger="backoff"
        ):
            func()

    backoff_re = re.compile("backing off", re.IGNORECASE)
    giveup_re = re.compile("giving up", re.IGNORECASE)

    backoff_log_count = 0
    giveup_log_count = 0
    for logger_name, level, message in caplog.record_tuples:
        if level == backoff_log_level and backoff_re.match(message):
            backoff_log_count += 1
        elif level == giveup_log_level and giveup_re.match(message):
            giveup_log_count += 1

    assert backoff_log_count == max_tries - 1
    assert giveup_log_count == 1


start = 0
elapsed = 0


def patch_sleep(n):
    global elapsed
    elapsed += n


def now():
    return start + elapsed


@pytest.fixture
def max_time_setup(monkeypatch):
    global start, elapsed
    start = datetime.datetime(2018, 1, 1, 12, 0, 10, 5).timestamp()
    elapsed = 0

    monkeypatch.setattr('time.sleep', patch_sleep)
    monkeypatch.setattr('timeit.default_timer', now)
    yield


def test_on_predicate_max_time_one_attempt_on_time(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        raise AssertionError("Invalid operation")

    def success(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 2

    # function succeeds on the first try, within allowed time
    @backoff.on_predicate(backoff.constant, interval=2, jitter=None, max_time=3,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(2)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 0)
    assert ret is True
    assert len(log) == 1


def test_on_predicate_max_time_two_attempts_on_time(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [2]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        raise AssertionError("Invalid operation")

    def success(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 5

    # function succeeds on the second try, within allowed time
    @backoff.on_predicate(backoff.constant, interval=1, jitter=None, max_time=6,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(2)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 1)
    assert ret is True
    assert len(log) == 2


def test_on_predicate_max_time_two_attempts_on_time_negative_interval(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [2]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        raise AssertionError("Invalid operation")

    def success(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 4

    # function succeeds on the second try, within allowed time
    @backoff.on_predicate(backoff.constant, interval=-1, jitter=None, max_time=6,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(2)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 1)
    assert ret is True
    assert len(log) == 2


def test_on_predicate_max_time_one_attempt_timeout(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 4

    def success(details):
        raise AssertionError("Invalid operation")

    # function fails on the first run and exceeds allowed time
    @backoff.on_predicate(backoff.constant, interval=2, jitter=None, max_time=3,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(4)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 10)
    assert ret is False
    assert len(log) == 1


def test_on_predicate_max_time_two_attempts_timeout(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [3]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 7

    def success(details, result):
        raise AssertionError("Invalid operation")

    # function fails twice, the completion of the second run is after allowed time
    @backoff.on_predicate(backoff.constant, interval=1, jitter=None, max_time=6,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(3)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 10)
    assert ret is False
    assert len(log) == 2


def test_on_predicate_max_time_two_attempts_timeout_callable(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [3]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 7

    def success(details, result):
        raise AssertionError("Invalid operation")

    def max_time():
        return 6

    # uses a function to provide max_time value
    @backoff.on_predicate(backoff.constant, interval=1, jitter=None, max_time=max_time,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(3)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 10)
    assert ret is False
    assert len(log) == 2


def test_on_predicate_max_time_one_attempt_zero_limit(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 4

    def success(details):
        raise AssertionError("Invalid operation")

    # function is executed once, even if max_time is set to zero
    @backoff.on_predicate(backoff.constant, interval=2, jitter=None, max_time=0,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(4)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 10)
    assert ret is False
    assert len(log) == 1


def test_on_predicate_max_time_one_attempt_long_interval(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 3

    def success(details):
        raise AssertionError("Invalid operation")

    # first attempt finishes before max_time,
    # but the remaining time till max_time is less than specified interval
    @backoff.on_predicate(backoff.constant, interval=2, jitter=None, max_time=4,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def return_true(log, n):
        patch_sleep(3)
        val = (len(log) == n)
        log.append(val)
        return val

    log = []
    ret = return_true(log, 10)
    assert ret is False
    assert len(log) == 1


def test_on_exception_max_time_one_attempt_on_time(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        raise AssertionError("Invalid operation")

    def success(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 2

    # function succeeds on the first try, within allowed time
    @backoff.on_exception(backoff.constant, KeyError, interval=2, jitter=None, max_time=3,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(2)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    ret = keyerror_then_true(log, 0)

    assert ret is True
    assert len(log) == 0


def test_on_exception_max_time_two_attempts_on_time(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [2]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        raise AssertionError("Invalid operation")

    def success(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 5

    # function succeeds on the second try, within allowed time
    @backoff.on_exception(backoff.constant, KeyError, interval=1, jitter=None, max_time=6,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(2)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    ret = keyerror_then_true(log, 1)
    assert ret is True
    assert len(log) == 1


def test_on_exception_max_time_two_attempts_on_time_negative_interval(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [2]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        raise AssertionError("Invalid operation")

    def success(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 4

    # function succeeds on the second try, within allowed time
    @backoff.on_exception(backoff.constant, KeyError, interval=-1, jitter=None,
                          max_time=6,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(2)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    ret = keyerror_then_true(log, 1)
    assert ret is True
    assert len(log) == 1


def test_on_exception_max_time_one_attempt_timeout(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 4

    def success(details):
        raise AssertionError("Invalid operation")

    # function fails on the first run and exceeds allowed time
    @backoff.on_exception(backoff.constant, KeyError, interval=2, jitter=None, max_time=3,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(4)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10)
    assert len(log) == 1


def test_on_exception_max_time_two_attempts_timeout(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [3]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 7

    def success(details, result):
        raise AssertionError("Invalid operation")

    # function fails twice, the completion of the second run is after allowed time
    @backoff.on_exception(backoff.constant, KeyError, interval=1, jitter=None, max_time=6,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(3)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10)
    assert len(log) == 2


def test_on_exception_max_time_two_attempts_timeout_callable(max_time_setup):

    def retry(details):
        count = details['tries']
        timestamps = [3]
        assert count == 1
        assert details['elapsed'] == timestamps[count - 1]

    def giveup(details):
        assert details['tries'] == 2
        assert details['elapsed'] == 7

    def success(details, result):
        raise AssertionError("Invalid operation")

    def max_time():
        return 6

    # uses a function to provide max_time value
    @backoff.on_exception(backoff.constant, KeyError, interval=1, jitter=None,
                          max_time=max_time,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(3)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10)
    assert len(log) == 2


def test_on_exception_max_time_one_attempt_zero_limit(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 4

    def success(details):
        raise AssertionError("Invalid operation")

    # function is executed once, even if max_time is set to zero
    @backoff.on_exception(backoff.constant, KeyError, interval=2, jitter=None, max_time=0,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(4)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10)
    assert len(log) == 1


def test_on_exception_max_time_one_attempt_long_interval(max_time_setup):

    def retry(details):
        raise AssertionError("Invalid operation")

    def giveup(details):
        assert details['tries'] == 1
        assert details['elapsed'] == 3

    def success(details):
        raise AssertionError("Invalid operation")

    # first attempt finishes before max_time,
    # but the remaining time till max_time is less than specified interval
    @backoff.on_exception(backoff.constant, KeyError, interval=2, jitter=None, max_time=4,
                          on_giveup=giveup, on_success=success, on_backoff=retry)
    def keyerror_then_true(log, n):
        patch_sleep(3)
        if len(log) == n:
            return True
        e = KeyError()
        log.append(e)
        raise e

    log = []
    with pytest.raises(KeyError):
        keyerror_then_true(log, 10)
    assert len(log) == 1
