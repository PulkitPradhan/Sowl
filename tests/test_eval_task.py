"""Tests for task definitions, schema validation, holdout split, and quick selection."""

from pathlib import Path

from ag.eval.task import load_tasks


def test_eval_tasks_loading_and_schema() -> None:
    tasks_dir = Path(__file__).resolve().parent.parent / "evals" / "tasks"
    tasks = load_tasks(tasks_dir=tasks_dir, quick=False)
    
    # Must have 30 total tasks
    assert len(tasks) == 30
    
    categories = {t.category for t in tasks}
    assert categories == {"normal", "multi_step", "ambiguous", "adversarial", "impossible"}

    # Counts per category
    cat_counts = {}
    for t in tasks:
        cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
    assert cat_counts["normal"] == 14
    assert cat_counts["multi_step"] == 6
    assert cat_counts["ambiguous"] == 4
    assert cat_counts["adversarial"] == 3
    assert cat_counts["impossible"] == 3

    # Holdout split check: >= 12 holdout tasks
    holdout_tasks = [t for t in tasks if t.split == "holdout"]
    dev_tasks = [t for t in tasks if t.split == "dev"]
    assert len(holdout_tasks) >= 12
    assert len(dev_tasks) + len(holdout_tasks) == 30

    # Ensure "Book me a flight to Goa" is used instead of "Order a pizza"
    impossible_prompts = [t.prompt for t in tasks if t.category == "impossible"]
    assert any("flight to goa" in p.lower() for p in impossible_prompts)
    assert not any("order a pizza" in p.lower() or "order me a pizza" in p.lower() for p in impossible_prompts)


def test_eval_quick_selection_deterministic_20() -> None:
    tasks_dir = Path(__file__).resolve().parent.parent / "evals" / "tasks"
    tasks_full = load_tasks(tasks_dir=tasks_dir, quick=False)
    tasks_quick = load_tasks(tasks_dir=tasks_dir, quick=True)

    # Exactly 20 tasks
    assert len(tasks_quick) == 20
    assert len(tasks_quick) < len(tasks_full)

    # Every task in quick must be in full
    quick_ids = [t.id for t in tasks_quick]
    full_ids = [t.id for t in tasks_full]
    for qid in quick_ids:
        assert qid in full_ids

    # Calling it twice yields exact same list
    tasks_quick_2 = load_tasks(tasks_dir=tasks_dir, quick=True)
    assert [t.id for t in tasks_quick_2] == quick_ids
