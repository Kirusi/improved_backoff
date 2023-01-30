import improved_backoff as backoff


def test_next_wait_constant_positive():
    wait = backoff._common._init_wait_gen(backoff.constant, {"interval": 1})
    seconds = backoff._common._next_wait(wait, 0, None, 3, 5)
    assert seconds == 1


def test_next_wait_constant_negative():
    wait = backoff._common._init_wait_gen(backoff.constant, {"interval": -1})
    seconds = backoff._common._next_wait(wait, 0, None, 3, 5)
    assert seconds == 0


def test_next_wait_constant_elapsed_exceeds_maximum():
    wait = backoff._common._init_wait_gen(backoff.constant, {"interval": 1})
    seconds = backoff._common._next_wait(wait, 0, None, 5, 3)
    assert seconds == 0


def test_next_wait_constant_with_jitter():

    def jitter(val):
        return val + 0.25

    wait = backoff._common._init_wait_gen(backoff.constant, {"interval": 1})
    seconds = backoff._common._next_wait(wait, 0, jitter, 3, 5)
    assert seconds == 1.25
