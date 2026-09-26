from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SampleRecord:
    id: str
    dataset: str
    context: str
    candidate: str
    reference: str | None = None
    label: float | int | str | None = None
    metadata: dict[str, str | float | int | bool | None] = field(default_factory=dict)
