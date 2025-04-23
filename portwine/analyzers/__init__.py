# from portwine.analyzers.base import Analyzer
# from portwine.analyzers.equitydrawdown import EquityDrawdownAnalyzer
# from portwine.analyzers.gridequitydrawdown import GridEquityDrawdownAnalyzer
# from portwine.analyzers.montecarlo import MonteCarloAnalyzer
# from portwine.analyzers.seasonality import SeasonalityAnalyzer
# from portwine.analyzers.correlation import CorrelationAnalyzer
# from portwine.analyzers.traintest import TrainTestEquityDrawdownAnalyzer
# from portwine.analyzers.strategycomparison import StrategyComparisonAnalyzer
# from portwine.analyzers.studentttest import StudentsTTestAnalyzer
# from portwine.analyzers.downsidecorrelation import DownsideCorrelationAnalyzer

# portwine/analyzers/__init__.py
from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import List, Dict, Type

from .base import Analyzer        # <-- the abstract base class for all analyzers


__all__: List[str] = []            # names we re‑export (populated below)
__modules_scanned__: Dict[str, ModuleType] = {}   # cache of imported modules


def _is_concrete_analyzer(obj: object) -> bool:
    """
    True if *obj* is a non‑abstract subclass of ``Analyzer``.
    """
    return (
        inspect.isclass(obj)
        and issubclass(obj, Analyzer)
        and obj is not Analyzer
        and not getattr(obj, "__abstractmethods__", False)
    )


def _eager_scan() -> None:
    """
    Import every sibling module once and hoist its concrete ``Analyzer``
    subclasses into this package’s namespace.
    """
    pkg_prefix = __name__ + "."
    for modinfo in pkgutil.walk_packages(__path__, pkg_prefix):
        if modinfo.ispkg:
            continue                                   # skip nested packages
        module = importlib.import_module(modinfo.name)
        __modules_scanned__[modinfo.name] = module

        for name, obj in inspect.getmembers(module, _is_concrete_analyzer):
            globals()[name] = obj                      # re‑export the class
            __all__.append(name)


_eager_scan()
