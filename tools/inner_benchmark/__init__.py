"""Inner-tuner benchmark foundation (PLAN §3, §十).

Flat-module package following the tools/tuners convention: importers put
``tools/inner_benchmark`` on sys.path and import the modules directly
(``import space``, ``import codec``, ``import state``, ``import artifacts``).
stdlib + numpy only at this layer — no optuna/torch imports.
"""
