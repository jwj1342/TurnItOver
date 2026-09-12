"""AssetSource protocol: anything that yields AssetSpecs (toy, PartNet-Mobility converter, ...)."""
from __future__ import annotations

from typing import Iterator, Protocol

from turnitover.core.spec import AssetSpec


class AssetSource(Protocol):
    source_id: str

    def __iter__(self) -> Iterator[AssetSpec]: ...

    def get(self, asset_id: str) -> AssetSpec: ...


def make_source(kind: str, **kwargs) -> AssetSource:
    if kind == "toy":
        from turnitover.assets.toy import ToyAssetSource

        return ToyAssetSource()
    if kind == "catalog":
        from turnitover.assets.catalog import CatalogSource

        return CatalogSource(**kwargs)
    raise ValueError(f"unknown asset source kind {kind!r}")
