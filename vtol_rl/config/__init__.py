import json
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Any, Optional

try:
    # Python 3.9+
    from importlib.resources import files, as_file
except Exception:  # pragma: no cover
    # <= Python 3.8
    from importlib_resources import files, as_file  # type: ignore


ROOT_PKG = __name__


@dataclass(frozen=True)
class JsonResource:
    """Represents a JSON resource (relative path to the config directory)"""

    relpath: str

    def load(self, *, encoding: str = "utf-8") -> Any:
        """Load and return the JSON content"""
        resource = files(ROOT_PKG).joinpath(self.relpath)
        if not resource.is_file():
            raise FileNotFoundError(f"JSON not found: {self.relpath}")
        with as_file(resource) as p:
            with open(p, "r", encoding=encoding) as f:
                return json.load(f)


def iter_jsons(*, under: Optional[str] = None) -> Iterator[Tuple[str, JsonResource]]:
    """
    Iterate over all *.json files under config (or under a specified subdirectory),
    yielding (relpath, JsonResource). relpath is relative to config, e.g. 'submoduleA/a1.json'.
    """
    root = files(ROOT_PKG)
    base = root.joinpath(under) if under else root
    for p in base.rglob("*.json"):
        # Only collect files recursively (rglob is recursive)
        rel = str(p.relative_to(root)).replace("\\", "/")
        yield rel, JsonResource(rel)


def load_json(relpath: str, *, encoding: str = "utf-8") -> Any:
    """
    Load a JSON file by relative path (relative to config root), e.g.:
        load_json("submoduleA/a1.json")
    """
    return JsonResource(relpath).load(encoding=encoding)


def load_all(*, under: Optional[str] = None, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Load all (or all under a specified subdirectory) JSON files at once, return {relpath: data}
    """
    out: Dict[str, Any] = {}
    for rel, res in iter_jsons(under=under):
        out[rel] = res.load(encoding=encoding)
    return out


def load_grouped_by_topdir(*, encoding: str = "utf-8") -> Dict[str, Dict[str, Any]]:
    """
    Load and group JSON files by top-level subdirectory under config, e.g. returns:
    {
        "submoduleA": {"a1.json": {...}, "a2.json": {...}},
        "submoduleB": {"b1.json": {...}},
    }
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for rel, res in iter_jsons():
        parts = rel.split("/", 1)
        top = parts[0]
        leaf = parts[1] if len(parts) > 1 else parts[0]
        grouped.setdefault(top, {})[leaf] = res.load(encoding=encoding)
    return grouped
