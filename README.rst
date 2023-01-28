backoff
=======

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://www.python.org/downloads/release/python-370
.. image:: https://img.shields.io/badge/python-3.8-blue.svg
    :target: https://www.python.org/downloads/release/python-380
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
    :target: https://www.python.org/downloads/release/python-390
.. image:: https://img.shields.io/badge/python-3.10-blue.svg
    :target: https://www.python.org/downloads/release/python-3100
.. image:: https://github.com/Kirusi/improved_backoff/workflows/Tests/badge.svg
    :target: https://github.com/Kirusi/improved_backoff/actions/workflows/tests.yml
.. image:: https://kirusi.github.io/improved_backoff/coverage.svg
    :target: https://github.com/Kirusi/improved_backoff/actions/workflows/coverage.yml
.. image:: https://img.shields.io/pypi/v/improved_backoff.svg
    :target: https://pypi.python.org/pypi/improved_backoff
.. image:: https://img.shields.io/github/license/kirusi/improved_backoff
    :target: https://github.com/kirusi/improved_backoff/blob/master/LICENSE

**Function decoration for backoff and retry**

This is a fork of an excellent Python library 
`backoff <https://github.com/litl/backoff>`_. This version includes 2 PRs
proposed in the original repo: `Correct check for max_time parameter <https://github.com/litl/backoff/pull/130>`_
and `Using timeit module for time management <https://github.com/litl/backoff/pull/185>`

In order to use this module import it under ``backoff`` alias and use it
the same way as the original module

.. code-block:: python
    import improved_backoff as backoff
