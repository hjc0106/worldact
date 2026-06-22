from __future__ import annotations

from worldact.data.objects import ObjectMemoryState


class ObjectMemory:
    def __init__(self) -> None:
        self._items: dict[str, ObjectMemoryState] = {}

    def add(self, state: ObjectMemoryState) -> None:
        key = state.name.strip().lower()
        if key in self._items:
            self._items[key].count += 1
        else:
            self._items[key] = state

    def get_all(self) -> list[ObjectMemoryState]:
        return list(self._items.values())
