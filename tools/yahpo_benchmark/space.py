from __future__ import annotations

import json
from typing import Any


def _plain(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


class SearchSpace:
    """Conditional ConfigSpace plus a rectangular HEBO representation."""

    def __init__(self, config_space, *, instance_name: str, instance: str):
        self.config_space = config_space
        self.instance_name = instance_name
        self.instance = instance
        self.hyperparameters = [
            hp
            for hp in config_space.get_hyperparameters()
            if hp.name != instance_name and hp.__class__.__name__ != "Constant"
        ]
        self.names = tuple(hp.name for hp in self.hyperparameters)
        self._by_name = {hp.name: hp for hp in self.hyperparameters}

    @property
    def dimension_count(self) -> int:
        return len(self.hyperparameters)

    def sample(self, count: int) -> list[dict[str, Any]]:
        sampled = self.config_space.sample_configuration(size=count)
        if count == 1:
            sampled = [sampled]
        return [self._strip_instance(cfg.get_dictionary()) for cfg in sampled]

    def canonicalize(self, params: dict[str, Any]) -> dict[str, Any]:
        from ConfigSpace import Configuration

        unknown = sorted(set(params) - set(self.names))
        if unknown:
            raise ValueError(f"unknown hyperparameters: {unknown}")
        values = {
            name: self._cast(self._by_name[name], value)
            for name, value in params.items()
            if value is not None
        }
        values[self.instance_name] = self._instance_value()
        cfg = Configuration(
            self.config_space,
            values=values,
            allow_inactive_with_values=False,
        )
        return self._strip_instance(cfg.get_dictionary())

    def encode_for_hebo(self, params: dict[str, Any]) -> dict[str, Any]:
        canonical = self.canonicalize(params)
        return {
            hp.name: _plain(canonical.get(hp.name, hp.default_value))
            for hp in self.hyperparameters
        }

    def decode_from_hebo(self, params: dict[str, Any]) -> dict[str, Any]:
        from ConfigSpace.util import deactivate_inactive_hyperparameters

        complete = {
            name: self._cast(self._by_name[name], params[name]) for name in self.names
        }
        complete[self.instance_name] = self._instance_value()
        cfg = deactivate_inactive_hyperparameters(
            configuration=complete,
            configuration_space=self.config_space,
        )
        return self._strip_instance(cfg.get_dictionary())

    def key(self, params: dict[str, Any]) -> str:
        canonical = self.canonicalize(params)
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"))

    def hebo_contract(self) -> dict[str, list]:
        contract: dict[str, list] = {}
        for hp in self.hyperparameters:
            kind = hp.__class__.__name__
            if "Float" in kind:
                item = ["float", float(hp.lower), float(hp.upper)]
                if bool(getattr(hp, "log", False)):
                    item.append("log")
            elif "Integer" in kind:
                item = ["int", int(hp.lower), int(hp.upper)]
            elif "Categorical" in kind:
                item = ["categorical", [_plain(v) for v in hp.choices]]
            elif "Ordinal" in kind:
                item = ["categorical", [_plain(v) for v in hp.sequence]]
            else:
                raise TypeError(f"unsupported ConfigSpace hyperparameter: {kind}")
            contract[hp.name] = item
        return contract

    def task_dimensions(self) -> list[dict[str, Any]]:
        dimensions = []
        for hp in self.hyperparameters:
            kind = hp.__class__.__name__
            item: dict[str, Any] = {
                "name": hp.name,
                "default": _plain(hp.default_value),
            }
            if "Float" in kind:
                item.update(
                    type="float",
                    lower=float(hp.lower),
                    upper=float(hp.upper),
                    log=bool(getattr(hp, "log", False)),
                )
            elif "Integer" in kind:
                item.update(
                    type="integer",
                    lower=int(hp.lower),
                    upper=int(hp.upper),
                    log=bool(getattr(hp, "log", False)),
                )
            elif "Categorical" in kind:
                item.update(type="categorical", choices=[_plain(v) for v in hp.choices])
            elif "Ordinal" in kind:
                item.update(type="ordinal", choices=[_plain(v) for v in hp.sequence])
            else:
                raise TypeError(f"unsupported ConfigSpace hyperparameter: {kind}")
            dimensions.append(item)
        return dimensions

    def conditions(self) -> list[str]:
        return [str(condition) for condition in self.config_space.get_conditions()]

    def _strip_instance(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            name: _plain(value)
            for name, value in values.items()
            if name != self.instance_name and name in self._by_name
        }

    def _instance_value(self) -> Any:
        hp = self.config_space.get_hyperparameter(self.instance_name)
        return self._cast(hp, self.instance)

    @staticmethod
    def _cast(hp, value: Any) -> Any:
        kind = hp.__class__.__name__
        if "Integer" in kind:
            return int(value)
        if "Float" in kind:
            return float(value)
        choices = getattr(hp, "choices", None)
        if choices is None:
            choices = getattr(hp, "sequence", None)
        if choices is not None:
            for choice in choices:
                if value == choice or str(value) == str(choice):
                    return _plain(choice)
        if kind == "Constant":
            constant = getattr(hp, "value")
            if value == constant or str(value) == str(constant):
                return _plain(constant)
        return _plain(value)
