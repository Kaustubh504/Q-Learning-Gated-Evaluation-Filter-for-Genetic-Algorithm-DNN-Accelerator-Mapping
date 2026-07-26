"""
q_filter.py
============
Q-Learning filter for GAMMA's genetic algorithm.

Maintains a Q-table keyed on genome state = per-level (sp_dim, sp_sz_bucket,
loop_order_tuple), for every cluster level in the genome.

Before MAESTRO is called for a candidate genome, the filter decides:
    action=1 → evaluate (call MAESTRO)
    action=0 → skip    (return cached -Inf, no MAESTRO call)

Update rule (no next-state, no gamma), applied only to genuinely-valid
MAESTRO results — infeasible/rejected genomes are tracked as a separate
per-state feasibility counter, never blended into the reward EMA:
    Q(s, 1) ← Q(s, 1) + α * [reward - Q(s, 1)]

Decision rule (ε-greedy):
    with prob ε     → always evaluate (explore)
    with prob (1-ε) → follow Q-table  (exploit), but only once a state has
                       been sampled >= min_samples times. Below that, always
                       evaluate — a single sample must never be trusted
                       enough to permanently gate a whole genome family.
        - if the state has been (almost) always infeasible → skip
          (pure feasibility pruning, saves MAESTRO calls on structurally
          broken regions of the search space)
        - otherwise, skip only if this state's Q-value is worse than the
          best reward seen so far by more than `relative_margin` of the
          best reward's magnitude. This threshold is RELATIVE to the
          current run's own reward scale (latency/energy/area/power all
          have wildly different magnitudes across models and layers), so
          it self-calibrates instead of relying on a hand-tuned constant.

The Q-table persists across layers and generations within one GAMMA run.
It can also be saved/loaded across runs for warm-starting.
"""

import random
import json
import os
import math


class QFilter:
    def __init__(
        self,
        alpha=0.1,             # learning rate
        epsilon=1.0,           # initial exploration rate (1.0 = evaluate everything at first)
        epsilon_decay=0.9,     # multiply epsilon by this after each generation
        epsilon_min=0.10,      # floor for epsilon (never go fully greedy)
        min_samples=5,         # a state needs this many observations before it can be skipped
        relative_margin=1.0,   # skip if Q(s) is worse than best-seen reward by more than this
                               # fraction of |best-seen reward| (self-calibrating, not a magic constant)
        infeasible_skip_rate=0.9,  # skip states that are infeasible at least this often
        q_table_path=None,    # optional path to save/load Q-table across runs
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.min_samples = min_samples
        self.relative_margin = relative_margin
        self.infeasible_skip_rate = infeasible_skip_rate
        self.q_table_path = q_table_path

        # Q-table: { state_key : float }  — EMA of *valid* rewards only
        self.q_table = {}
        # Per-state sample counts, tracked separately from the reward signal
        # so a run of infeasible genomes can't drag down the quality estimate
        # of an otherwise-good state.
        self.valid_counts = {}
        self.invalid_counts = {}
        # Best valid reward seen so far across all states (this run). Used
        # as the reference point for the relative skip threshold.
        self.best_reward_seen = None

        # Stats per generation for logging
        self.gen_stats = []   # list of dicts: {gen, evaluated, skipped, epsilon}

        if q_table_path and os.path.exists(q_table_path):
            self.load(q_table_path)
            print(f"[QFilter] Loaded Q-table from {q_table_path} ({len(self.q_table)} states)")

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------
    def extract_state(self, indv):
        """
        Extract a compact hashable state from a genome individual.

        State = tuple of (sp_dim, sp_sz_bucket, loop_order) for EVERY
        cluster level present in the genome (not just the first one).

        - sp_dim / loop_order captures parallelization dimension and data
          reuse ordering, per level.
        - sp_sz_bucket is a coarse log2 bucket of the spatial cluster size
          — cheap to add, and correlates strongly with feasibility/area,
          without blowing up the state space the way raw tile sizes would.
        - Exact tile sizes are still excluded on purpose: that's the
          continuous dimension the GA's mutation/crossover explores most,
          and we want genomes that differ only in tile size to still share
          Q-table entries (generalization), just not collapse levels or
          cluster-size class entirely like the previous single-level state did.
        """
        if len(indv) < 7:
            return (("UNKNOWN", 0, ()),)
        num_levels = len(indv) // 7
        levels = []
        for lvl in range(num_levels):
            offset = lvl * 7
            sp_dim = indv[offset][0]
            sp_sz = indv[offset][1]
            loop_order = tuple(indv[offset + i][0] for i in range(1, 7))
            size_bucket = int(math.log2(max(1, sp_sz)))
            levels.append((sp_dim, size_bucket, loop_order))
        return tuple(levels)

    def _state_key(self, state):
        """Convert state tuple to a JSON-serializable string key."""
        parts = []
        for sp_dim, size_bucket, loop_order in state:
            parts.append(f"{sp_dim}{size_bucket}|{''.join(loop_order)}")
        return "||".join(parts)

    # ------------------------------------------------------------------
    # Core Q-table operations
    # ------------------------------------------------------------------
    def get_q_value(self, state):
        """Return Q(s, action=1). Default = 0.0 (neutral, unknown state)."""
        key = self._state_key(state)
        return self.q_table.get(key, 0.0)

    def update(self, state, reward=None, valid=True):
        """
        Update statistics for this state after receiving a real MAESTRO result.

        Args:
            state:  extracted state tuple
            reward: float — fitness1 value from MAESTRO. Ignored if valid=False.
            valid:  whether MAESTRO actually accepted this genome. Invalid
                    genomes only bump the infeasibility counter — they must
                    NOT be blended into the reward EMA (that was the bug:
                    folding a -1e6 sentinel into the same average as real
                    rewards drags down states that are simply near the
                    feasibility boundary, which is often exactly where the
                    best solutions live).
        """
        key = self._state_key(state)
        if not valid:
            self.invalid_counts[key] = self.invalid_counts.get(key, 0) + 1
            return

        self.valid_counts[key] = self.valid_counts.get(key, 0) + 1
        # Bootstrap at the first observed reward instead of an artificial
        # 0.0 baseline — EMA-ing from 0 biases early estimates toward an
        # unrealistic "neutral" value for however many samples it takes
        # alpha to wash it out.
        old_q = self.q_table.get(key, reward)
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[key] = new_q

        if self.best_reward_seen is None or reward > self.best_reward_seen:
            self.best_reward_seen = reward

    # ------------------------------------------------------------------
    # Decision: should we evaluate this genome?
    # ------------------------------------------------------------------
    def should_evaluate(self, indv):
        """
        ε-greedy decision for one candidate genome.

        Returns:
            True  → call MAESTRO (evaluate)
            False → skip (don't call MAESTRO)
        """
        if random.random() < self.epsilon:
            return True   # explore: always evaluate

        state = self.extract_state(indv)
        key = self._state_key(state)
        n_valid = self.valid_counts.get(key, 0)
        n_invalid = self.invalid_counts.get(key, 0)
        n_total = n_valid + n_invalid

        # Not enough evidence yet — never gate on a handful of samples.
        if n_total < self.min_samples:
            return True

        # Pure feasibility pruning: this state is (almost) always rejected
        # by MAESTRO regardless of tile sizes.
        invalid_rate = n_invalid / n_total
        if invalid_rate >= self.infeasible_skip_rate and n_valid == 0:
            return False

        if n_valid == 0 or self.best_reward_seen is None:
            return True

        q_val = self.q_table.get(key, self.best_reward_seen)
        gap = self.best_reward_seen - q_val  # >= 0: how much worse than best
        threshold = self.relative_margin * abs(self.best_reward_seen)
        return gap <= threshold

    # ------------------------------------------------------------------
    # After each generation: decay epsilon, log stats
    # ------------------------------------------------------------------
    def end_of_generation(self, gen, n_evaluated, n_skipped):
        """
        Call this once per generation after all evaluations are done.
        Decays epsilon and logs stats.

        Args:
            gen:         generation index (0-based)
            n_evaluated: how many genomes were actually sent to MAESTRO
            n_skipped:   how many genomes were filtered out by Q-table
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        stat = {
            "gen":        gen + 1,
            "evaluated":  n_evaluated,
            "skipped":    n_skipped,
            "epsilon":    round(self.epsilon, 4),
            "q_states":   len(self.q_table),
        }
        self.gen_stats.append(stat)
        print(
            f"[QFilter] Gen {gen+1}: evaluated={n_evaluated}, skipped={n_skipped}, "
            f"ε={self.epsilon:.3f}, Q-states known={len(self.q_table)}"
        )

    # ------------------------------------------------------------------
    # Save / load Q-table across runs (warm-starting)
    # ------------------------------------------------------------------
    def save(self, path=None):
        path = path or self.q_table_path
        if path is None:
            return
        with open(path, "w") as f:
            json.dump({
                "q_table": self.q_table,
                "valid_counts": self.valid_counts,
                "invalid_counts": self.invalid_counts,
                "best_reward_seen": self.best_reward_seen,
            }, f, indent=2)
        print(f"[QFilter] Q-table saved → {path} ({len(self.q_table)} states)")

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        if "q_table" in data:
            self.q_table = data["q_table"]
            self.valid_counts = data.get("valid_counts", {})
            self.invalid_counts = data.get("invalid_counts", {})
            self.best_reward_seen = data.get("best_reward_seen", None)
        else:
            # backward-compat with old flat-dict Q-table files
            self.q_table = data

    # ------------------------------------------------------------------
    # Summary after full run
    # ------------------------------------------------------------------
    def print_summary(self):
        if not self.gen_stats:
            return
        total_eval  = sum(s["evaluated"] for s in self.gen_stats)
        total_skip  = sum(s["skipped"]   for s in self.gen_stats)
        total       = total_eval + total_skip
        skip_pct    = (total_skip / total * 100) if total > 0 else 0
        print("\n" + "="*60)
        print("[QFilter] Run Summary")
        print(f"  Total candidates generated : {total}")
        print(f"  Total MAESTRO calls        : {total_eval}")
        print(f"  Total skipped by Q-filter  : {total_skip}  ({skip_pct:.1f}%)")
        print(f"  Final ε                    : {self.epsilon:.4f}")
        print(f"  Unique states in Q-table   : {len(self.q_table)}")
        print("="*60)
        # Top 5 best states
        if self.q_table:
            sorted_states = sorted(self.q_table.items(), key=lambda x: x[1], reverse=True)
            print("  Top 5 genome states by Q-value:")
            for k, v in sorted_states[:5]:
                print(f"    {k:30s}  Q={v:.2f}")
        print("="*60 + "\n")
