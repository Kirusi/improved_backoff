improved\_backoff
=================

[![image](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370)
[![image](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380)
[![image](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390)
[![image](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100)
[![image](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110)

[![image](https://github.com/Kirusi/improved_backoff/workflows/tests/badge.svg)](https://github.com/Kirusi/improved_backoff/actions/workflows/tests.yml)
[![image](https://kirusi.github.io/improved_backoff/coverage.svg)](https://github.com/Kirusi/improved_backoff/actions/workflows/coverage.yml)
[![image](https://img.shields.io/pypi/v/improved_backoff.svg)](https://pypi.python.org/pypi/improved_backoff)
[![image](https://img.shields.io/github/license/kirusi/improved_backoff)](https://github.com/kirusi/improved_backoff/blob/master/LICENSE)

**Function decoration for backoff and retry**

This is a fork of an excellent Python library
[backoff](https://github.com/litl/backoff). The library was forked on January 26, 2023.
It includes the original version 2.2.1 (October 5, 2022) and a few extra updates.

The work in this repo was motivated by the following PRs that were proposed in the original repo:

-   [Correct check for max\_time parameter](https://github.com/litl/backoff/pull/130)
-   [Fix some issues around max\_time in \_next\_wait](https://github.com/litl/backoff/pull/136)
-   [Using \"timeit\" module for time management](https://github.com/litl/backoff/pull/185)

The updated behavior of `max_time` is that a function will be retried only if the elapsed time is less
than `max_time`. Also the remaining time should exceed the `interval` (delay between retries).

The code that calculates delay time was modified to not generate negative values.
If at the time of calculation elapsed time exceeds specified `max_time` value than delay is set to zero.
If calculated delay time is negative, then it's also set to zero.

In order to use this module import it under `backoff` alias and use it
the same way as the original module

```python
import improved_backoff as backoff

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def get_url(url):
    return requests.get(url)
```
