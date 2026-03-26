"""
Ebbinghaus Forgetting Curve Utility for MemoryBank-Baseline
============================================================
Class-based port of MemoryBank-SiliconFriend/memory_bank/memory_retrieval/forget_memory.py

The core class MemoryForgetter mirrors the MemoryForgetterLoader class from SiliconFriend
but is adapted to the Baseline's flat data format (no LangChain dependency).

Usage in run_inference.py:
    from forget_utility import MemoryForgetter

    forgetter = MemoryForgetter(memory_file=args.memory_file)
    forgetter.load_memories()

    # Before each conversation date:
    forgetter.apply_forgetting(cur_date="2024-06-01")

    # After retrieval step:
    forgetter.update_on_recall(retrieved_memos, cur_date="2024-06-01")

    # Persist at the end:
    forgetter.write_memories()
"""

import math
import datetime
import json


# ─────────────────────────────────────────────
# Standalone formula (mirrors SiliconFriend's forgetting_curve)
# ─────────────────────────────────────────────

def forgetting_curve(t: float, S: float) -> float:
    """
    Ebbinghaus retention formula from the paper: R = e^(-t / S)

    :param t: Days elapsed since the memory was last recalled.
    :param S: Memory strength (>=1; increments with each successful recall).
    :return:  Retention probability in [0, 1].
    """
    if S <= 0:
        S = 1
    return math.exp(-t / S)


# ─────────────────────────────────────────────
# MemoryForgetter class
# ─────────────────────────────────────────────

class MemoryForgetter:
    """
    Manages the Ebbinghaus Forgetting Curve for MemoryBank-Baseline.

    Mirrors MemoryForgetterLoader from SiliconFriend's forget_memory.py.

    Each dialogue entry in memory.json gets three extra fields (injected on
    first use if absent):
        - memory_strength    (int, default=1): Higher → slower forgetting.
        - last_recall_date   (str, YYYY-MM-DD): Date of most recent recall.
        - memory_id          (str): Unique ID, e.g. "P_001_2024-01-10_0".

    Attributes:
        memory_file  (str):  Path to memory.json.
        memory_bank  (dict): The loaded in-memory representation.
    """

    def __init__(self, memory_file: str, retention_threshold: float = 0.3):
        self.memory_file = memory_file
        self.retention_threshold = retention_threshold
        self.memory_bank: dict = {}

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_memories(self) -> None:
        """Load memory.json from disk into self.memory_bank."""
        with open(self.memory_file, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)
        print(f"  [MemoryForgetter] Loaded {len(self.memory_bank)} persona(s) from {self.memory_file}")

    def write_memories(self, out_file: str = None) -> None:
        """Persist self.memory_bank to disk (overwrites memory_file by default)."""
        target = out_file or self.memory_file
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.memory_bank, f, indent=2, ensure_ascii=False)
        print(f"  [MemoryForgetter] Saved updated memory → {target}")

    # ── Date helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _days_between(date_early: str, date_late: str) -> int:
        """Return (date_late - date_early) in days. Returns 0 for negative/invalid."""
        fmt = "%Y-%m-%d"
        try:
            d1 = datetime.datetime.strptime(date_early, fmt)
            d2 = datetime.datetime.strptime(date_late, fmt)
            return max(0, (d2 - d1).days)
        except ValueError:
            return 0

    # ── Metadata injection ────────────────────────────────────────────────────

    def _ensure_metadata(self) -> None:
        """
        Walk all history entries and inject forgetting metadata if absent.
        Mirrors SiliconFriend's initial_load_forget_and_save() metadata setup.

        Should be called once after load_memories().
        """
        for pid, persona_mem in self.memory_bank.items():
            for date, qa_pairs in persona_mem.get("history", {}).items():
                for i, entry in enumerate(qa_pairs):
                    # Handle legacy list format [query, response]
                    if isinstance(entry, list):
                        entry = {"query": entry[0], "response": entry[1]}
                        persona_mem["history"][date][i] = entry
                    entry.setdefault("memory_strength",  1)
                    entry.setdefault("last_recall_date", date)
                    entry.setdefault("memory_id",        f"{pid}_{date}_{i}")

            # Inject metadata into summaries too
            for date, summary in persona_mem.get("summary", {}).items():
                if isinstance(summary, dict):
                    summary.setdefault("memory_strength",  1)
                    summary.setdefault("last_recall_date", date)
                    summary.setdefault("memory_id",        f"{pid}_{date}_summary")

    # ── Core forgetting ───────────────────────────────────────────────────────

    def get_persona_memory(self, persona_id: str) -> dict:
        """Return the current in-memory state for a persona."""
        return self.memory_bank.get(persona_id, {})

    def get_active_memory_ids(self, persona_id: str, max_date: str = None) -> set:
        """
        Return surviving memory IDs for one persona, optionally restricted to dates
        strictly earlier than max_date.
        """
        self._ensure_metadata()

        active_ids = set()
        persona_mem = self.memory_bank.get(persona_id, {})

        for date, qa_pairs in persona_mem.get("history", {}).items():
            if max_date and date >= max_date:
                continue
            for idx, entry in enumerate(qa_pairs):
                active_ids.add(entry.get("memory_id", f"{persona_id}_{date}_{idx}"))

        for date, summary in persona_mem.get("summary", {}).items():
            if max_date and date >= max_date:
                continue
            if isinstance(summary, dict):
                active_ids.add(summary.get("memory_id", f"{persona_id}_{date}_summary"))

        return active_ids

    def apply_forgetting(self, cur_date: str, persona_id: str = None) -> None:
        """
        Apply the Ebbinghaus forgetting curve to self.memory_bank for cur_date.

        For every dialogue entry and summary:
          - Compute t = days since last_recall_date.
          - Compute retention = forgetting_curve(t, memory_strength).
          - Delete that entry with probability (1 - retention), matching the
            released SiliconFriend implementation.

        Mirrors SiliconFriend's initial_load_forget_and_save() forget logic.

        Args:
            cur_date (str): The current simulation date ("YYYY-MM-DD").
            persona_id (str): If provided, only update this persona.
        """
        # Ensure metadata exists before applying forgetting
        self._ensure_metadata()

        persona_items = (
            [(persona_id, self.memory_bank.get(persona_id, {}))]
            if persona_id else list(self.memory_bank.items())
        )

        for pid, persona_mem in persona_items:
            history = persona_mem.get("history", {})
            summary = persona_mem.get("summary", {})
            dates_to_remove = []

            for date, qa_pairs in list(history.items()):
                forget_indices = []

                for i, entry in enumerate(qa_pairs):
                    t = self._days_between(entry.get("last_recall_date", date), cur_date)
                    S = entry.get("memory_strength", 1)
                    retention = forgetting_curve(t, S)

                    if retention < self.retention_threshold:
                        forget_indices.append(i)
                        print(f"  [FORGET] {pid} | {date}[{i}] | "
                              f"t={t}d S={S} R={retention:.3f} → FORGOTTEN")
                    else:
                        print(f"  [KEEP]   {pid} | {date}[{i}] | "
                              f"t={t}d S={S} R={retention:.3f} → KEPT")

                # Remove forgotten entries (reverse order to preserve indices)
                for idx in sorted(forget_indices, reverse=True):
                    qa_pairs.pop(idx)

                # Date-level summaries become stale once any turn from that date is gone.
                if forget_indices and date in summary:
                    del summary[date]
                    print(f"  [FORGET] {pid} | summary[{date}] removed (source turns changed).")

                # If entire date is forgotten, mark for removal
                if not qa_pairs:
                    dates_to_remove.append(date)
                    if date in summary:
                        del summary[date]

            for date, summary_entry in list(summary.items()):
                if not isinstance(summary_entry, dict):
                    continue
                t = self._days_between(summary_entry.get("last_recall_date", date), cur_date)
                S = summary_entry.get("memory_strength", 1)
                retention = forgetting_curve(t, S)

                if retention < self.retention_threshold:
                    del summary[date]
                    print(f"  [FORGET] {pid} | summary[{date}] | "
                          f"t={t}d S={S} R={retention:.3f} → FORGOTTEN")
                else:
                    print(f"  [KEEP]   {pid} | summary[{date}] | "
                          f"t={t}d S={S} R={retention:.3f} → KEPT")

            for date in dates_to_remove:
                del history[date]
                print(f"  [FORGET] {pid} | date={date} entirely forgotten (all turns gone).")

    # ── Reinforcement ─────────────────────────────────────────────────────────

    def update_on_recall(self, retrieved_memos: list, cur_date: str, persona_id: str = None) -> None:
        """
        Reinforce memories that were successfully recalled by FAISS search.

        For each retrieved memory chunk, match on stable memory_id and increment
        memory_strength by 1, updating last_recall_date.

        Mirrors SiliconFriend's update_memory_when_searched().

        Args:
            retrieved_memos (list): Dicts from BERTMemoryRetrieval.search():
                                    [{"text": ..., "date": ..., "score": ..., "memory_id": ...}, ...]
            cur_date        (str):  Current date string "YYYY-MM-DD".
            persona_id      (str):  If provided, only update this persona.
        """
        self._ensure_metadata()

        persona_items = (
            [(persona_id, self.memory_bank.get(persona_id, {}))]
            if persona_id else list(self.memory_bank.items())
        )

        for memo in retrieved_memos:
            memory_id = memo.get("memory_id")
            if not memory_id:
                continue

            for pid, persona_mem in persona_items:
                matched = False

                for memo_date, entries in persona_mem.get("history", {}).items():
                    for idx, entry in enumerate(entries):
                        entry_id = entry.get("memory_id", f"{pid}_{memo_date}_{idx}")
                        if entry_id == memory_id:
                            old_s = entry.get("memory_strength", 1)
                            entry["memory_strength"] = old_s + 1
                            entry["last_recall_date"] = cur_date
                            print(f"  [REINFORCE] {pid} | {memo_date} | "
                                  f"strength: {old_s} → {entry['memory_strength']}")
                            matched = True
                            break
                    if matched:
                        break

                if matched:
                    break

                for memo_date, summary_entry in persona_mem.get("summary", {}).items():
                    if not isinstance(summary_entry, dict):
                        continue
                    summary_id = summary_entry.get("memory_id", f"{pid}_{memo_date}_summary")
                    if summary_id == memory_id:
                        old_s = summary_entry.get("memory_strength", 1)
                        summary_entry["memory_strength"] = old_s + 1
                        summary_entry["last_recall_date"] = cur_date
                        print(f"  [REINFORCE] {pid} | {memo_date} | "
                              f"strength: {old_s} → {summary_entry['memory_strength']}")
                        break
