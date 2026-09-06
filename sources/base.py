"""A Source yields Frames. That is the whole interface."""
from __future__ import annotations

import abc


class Source(abc.ABC):
    """Bound to one stream ("overhead" | "shelf") at startup by store.yaml."""

    stream: str
    fps: float

    @abc.abstractmethod
    def frames(self):
        """Yield Frame objects until exhausted or closed."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
