import signal

from inspect import ismethod


class DelayedKeyboardInterrupt(object):
    """Create an atomic section with respect to the Keyboard Interrupt.

    Notes
    -----
    When entering an "atomic" section we divert every sigint signal to our
    handler, which reraises it when the section is left.

    Typically used in the `with` statement in the following manner:
    >>> with DelayedKeyboardInterrupt() as flag:
    >>>     for i in range():
    >>>         ...
    >>>         if flag:
    >>>             break
    """

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __bool__(self):
        return self.signal is not None

    def __enter__(self):
        self.signal, self.is_nested_ = None, False

        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

        if ismethod(self.old_handler):
            self.is_nested_ = isinstance(
                self.old_handler.__self__, DelayedKeyboardInterrupt
            )

        return self

    def handler(self, sig, frame):
        self.signal = sig, frame

        if self.is_nested_:
            self.old_handler(sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal:
            self.old_handler(*self.signal)
