"""Task metadata and deterministic selection for LIBERO-Plus."""

from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


CATEGORY_SLUGS = {
    "Objects Layout": "objects_layout",
    "Camera Viewpoints": "camera_viewpoints",
    "Robot Initial States": "robot_initial_states",
    "Language Instructions": "language_instructions",
    "Light Conditions": "light_conditions",
    "Background Textures": "background_textures",
    "Sensor Noise": "sensor_noise",
}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


CATEGORY_ALIASES = {
    alias: canonical
    for canonical, slug in CATEGORY_SLUGS.items()
    for alias in {_normalize_name(canonical), slug}
}


@dataclass(frozen=True)
class PlusTask:
    suite: str
    task_id: int
    official_id: int
    name: str
    category: str
    category_slug: str
    difficulty_level: Optional[int]

    @property
    def difficulty_label(self) -> str:
        return "unknown" if self.difficulty_level is None else str(self.difficulty_level)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["difficulty_label"] = self.difficulty_label
        return value


@dataclass(frozen=True)
class PlusShard:
    shard_id: int
    suite: str
    tasks: tuple[PlusTask, ...]

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "suite": self.suite,
            "num_tasks": len(self.tasks),
            "task_ids": [task.task_id for task in self.tasks],
        }


def resolve_repo_path(project_root: Path, configured_path: Optional[str]) -> Path:
    path = Path(configured_path).expanduser() if configured_path else project_root / "LIBERO-plus"
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def classification_path(repo_path: Path) -> Path:
    return repo_path / "libero" / "libero" / "benchmark" / "task_classification.json"


def canonicalize_categories(categories: Optional[Iterable[str]]) -> Optional[set[str]]:
    if categories is None:
        return None
    canonical = set()
    invalid = []
    for category in categories:
        key = _normalize_name(str(category))
        if key not in CATEGORY_ALIASES:
            invalid.append(str(category))
        else:
            canonical.add(CATEGORY_ALIASES[key])
    if invalid:
        valid = ", ".join(CATEGORY_SLUGS.values())
        raise ValueError(f"Unknown LIBERO-Plus categories {invalid}. Valid category slugs: {valid}")
    return canonical


def load_tasks(repo_path: Path) -> list[PlusTask]:
    metadata_file = classification_path(repo_path)
    if not metadata_file.is_file():
        raise FileNotFoundError(f"LIBERO-Plus classification file not found: {metadata_file}")

    raw = json.loads(metadata_file.read_text(encoding="utf-8"))
    task_map_file = metadata_file.with_name("libero_suite_task_map.py")
    syntax_tree = ast.parse(task_map_file.read_text(encoding="utf-8"), filename=str(task_map_file))
    task_map_assignment = next(
        (
            node
            for node in syntax_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets)
        ),
        None,
    )
    if task_map_assignment is None:
        raise ValueError(f"Could not find libero_task_map in {task_map_file}")
    benchmark_task_map = ast.literal_eval(task_map_assignment.value)
    tasks = []
    for suite, rows in raw.items():
        benchmark_names = benchmark_task_map.get(suite)
        if benchmark_names is None or len(benchmark_names) != len(rows):
            raise ValueError(
                f"Classification/task-map size mismatch for {suite}: "
                f"classification={len(rows)}, task_map={None if benchmark_names is None else len(benchmark_names)}"
            )
        for task_id, row in enumerate(rows):
            official_id = int(row["id"])
            if official_id != task_id + 1:
                raise ValueError(
                    f"Non-sequential official ID in {suite}: row={task_id}, id={official_id}"
                )
            if row["name"] != benchmark_names[task_id]:
                raise ValueError(
                    f"Classification/task-map name mismatch in {suite} at task_id={task_id}."
                )
            category = str(row["category"])
            if category not in CATEGORY_SLUGS:
                raise ValueError(f"Unknown category in classification file: {category}")
            difficulty = row.get("difficulty_level")
            tasks.append(
                PlusTask(
                    suite=suite,
                    task_id=task_id,
                    official_id=official_id,
                    name=str(row["name"]),
                    category=category,
                    category_slug=CATEGORY_SLUGS[category],
                    difficulty_level=None if difficulty is None else int(difficulty),
                )
            )
    return tasks


def select_tasks(
    tasks: Iterable[PlusTask],
    *,
    suites: Iterable[str],
    categories: Optional[Iterable[str]] = None,
    difficulty_levels: Optional[Iterable[int]] = None,
    include_unknown_difficulty: bool = True,
    task_ids: Optional[Iterable[int]] = None,
    max_tasks_per_cell: Optional[int] = None,
    sample_seed: int = 0,
) -> list[PlusTask]:
    suites_set = {str(value) for value in suites}
    categories_set = canonicalize_categories(categories)
    difficulties_set = (
        None if difficulty_levels is None else {int(value) for value in difficulty_levels}
    )
    task_ids_set = None if task_ids is None else {int(value) for value in task_ids}

    selected = []
    for task in tasks:
        if task.suite not in suites_set:
            continue
        if categories_set is not None and task.category not in categories_set:
            continue
        if task_ids_set is not None and task.task_id not in task_ids_set:
            continue
        if task.difficulty_level is None:
            if not include_unknown_difficulty:
                continue
        elif difficulties_set is not None and task.difficulty_level not in difficulties_set:
            continue
        selected.append(task)

    if max_tasks_per_cell is not None:
        limit = int(max_tasks_per_cell)
        if limit <= 0:
            raise ValueError("LIBERO_PLUS.max_tasks_per_cell must be positive or null.")
        cells: dict[tuple[str, str, str], list[PlusTask]] = {}
        for task in selected:
            key = (task.suite, task.category_slug, task.difficulty_label)
            cells.setdefault(key, []).append(task)
        selected = []
        for key in sorted(cells):
            cell_tasks = sorted(cells[key], key=lambda item: item.task_id)
            if len(cell_tasks) > limit:
                rng = random.Random(f"{int(sample_seed)}:{':'.join(key)}")
                cell_tasks = rng.sample(cell_tasks, limit)
            selected.extend(cell_tasks)

    return sorted(selected, key=lambda item: (item.suite, item.task_id))


def write_task_files(tasks: Iterable[PlusTask], task_file: Path, manifest_file: Path) -> int:
    task_list = list(tasks)
    task_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    with task_file.open("w", encoding="utf-8") as handle:
        for task in task_list:
            handle.write(f"{task.suite},{task.task_id}\n")

    with manifest_file.open("w", encoding="utf-8") as handle:
        for task in task_list:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=True) + "\n")
    return len(task_list)


def create_balanced_shards(
    tasks: Iterable[PlusTask],
    tasks_per_shard: int,
) -> list[PlusShard]:
    shard_size = int(tasks_per_shard)
    if shard_size <= 0:
        raise ValueError("MULTIRUN.tasks_per_worker must be positive.")

    by_suite: dict[str, list[PlusTask]] = {}
    for task in sorted(tasks, key=lambda item: (item.suite, item.task_id)):
        by_suite.setdefault(task.suite, []).append(task)

    shards = []
    shard_id = 0
    for suite in sorted(by_suite):
        suite_tasks = by_suite[suite]
        num_shards = math.ceil(len(suite_tasks) / shard_size)
        buckets: list[list[PlusTask]] = [[] for _ in range(num_shards)]
        for index, task in enumerate(suite_tasks):
            buckets[index % num_shards].append(task)
        for bucket in buckets:
            shards.append(
                PlusShard(
                    shard_id=shard_id,
                    suite=suite,
                    tasks=tuple(bucket),
                )
            )
            shard_id += 1
    return shards


def write_shard_files(
    shards: Iterable[PlusShard],
    scheduler_task_file: Path,
    shard_dir: Path,
) -> int:
    shard_list = list(shards)
    scheduler_task_file.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    with scheduler_task_file.open("w", encoding="utf-8") as handle:
        for shard in shard_list:
            handle.write(f"libero_plus_shard,{shard.shard_id}\n")

    with (shard_dir / "index.jsonl").open("w", encoding="utf-8") as index_handle:
        for shard in shard_list:
            index_handle.write(json.dumps(shard.to_dict(), ensure_ascii=True) + "\n")
            shard_file = shard_dir / f"shard_{shard.shard_id:05d}.jsonl"
            with shard_file.open("w", encoding="utf-8") as shard_handle:
                for task in shard.tasks:
                    shard_handle.write(json.dumps(task.to_dict(), ensure_ascii=True) + "\n")
    return len(shard_list)


def read_manifest(path: Path) -> list[PlusTask]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                PlusTask(
                    suite=row["suite"],
                    task_id=int(row["task_id"]),
                    official_id=int(row["official_id"]),
                    name=row["name"],
                    category=row["category"],
                    category_slug=row["category_slug"],
                    difficulty_level=row.get("difficulty_level"),
                )
            )
    return tasks
