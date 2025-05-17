import yaml
from pathlib import Path
from typing import TypeVar, Type, Mapping
import pathlib, dataclasses 

T = TypeVar("T")

def load_config(path: str | Path,model_cls: Type[T] ,section: str = None) -> T:
    raw = yaml.safe_load(Path(path).read_text())
    data: Mapping = raw if section is None else raw.get(section)

    if data is None:
        raise KeyError(f"Section '{section}' not found in {path}")
    
    if not isinstance(data, Mapping):
        raise TypeError(
            f"Section '{section}' should be a mapping, got {type(data).__name__}"
        )
    if not dataclasses.is_dataclass(model_cls):
        raise TypeError(f"{model_cls} is not a dataclass")

    return model_cls(**data)  