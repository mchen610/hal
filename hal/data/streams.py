"""Named-stream registry for cloud-hosted MDS datasets.

Stub for Part 2 productionization (see TODO_PRODUCTIONIZATION.md). The
training config imports ``StreamRegistry`` at module load; this file
exists so that import works. Actual remote stream entries land when R2
hosting is wired up.

Until then, ``StreamRegistry.get(name)`` raises ``NotImplementedError``
on every name — training callers must point at ``data_dir`` (local MDS)
instead of ``streams``.
"""

from streaming import Stream


class StreamRegistry:
    """Registry of named remote streams.

    Today: empty. Use ``DataConfig.data_dir`` for local MDS shards.
    """

    _entries: dict[str, Stream] = {}

    @classmethod
    def get(cls, name: str) -> Stream:
        if name in cls._entries:
            return cls._entries[name]
        raise NotImplementedError(
            f"stream {name!r} not registered. Cloud-hosted streams are not "
            f"wired up yet (see TODO_PRODUCTIONIZATION.md). For now, point "
            f"DataConfig.data_dir at a local MDS directory."
        )

    @classmethod
    def register(cls, name: str, stream: Stream) -> None:
        cls._entries[name] = stream
