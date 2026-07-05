# solution_g++_alns_regret_mainline

Mainline comparison version copied from `solution_g++_alns_regret`.

Default behavior:

- Keeps the frozen small-window mainline defaults.
- Enables ALNS-style regret repair by default.
- Uses the train/validation sweet-spot config from `eval/configs/alns_regret/regret3_cap48.json`.

Frozen default values for this version:

- `regret_repair_enabled = 1`
- `regret_repair_k = 3`
- `regret_repair_weight = 2.0`
- `regret_repair_candidate_limit = 32`
- `regret_repair_scope = 0`
- `regret_repair_max_block = 48`
- `repair_rounds = 60`
- `small_window_repair_rounds = 20`
- `small_window_repair_cap = 64`

This is not the previously frozen submission candidate. It is a separate competitive mainline version that must be evaluated independently.
