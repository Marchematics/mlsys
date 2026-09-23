"""R112: redundancy-structured demonstration pools (new module; R110 untouched).

Motivation.  R110 measured ``H_state = (oracle_greedy - oracle_static) / oracle_greedy
<= 5e-4`` on all five benchmarks: with an "8 nearest + 8 distractors" pool the value
of a demonstration barely depends on what is already selected, so the family tested
ranking transfer but not the state-conditioning mechanism.  R112 keeps every other
protocol component identical and changes **only the pool construction**:

* K = 16 candidates, budget B = 4 (unchanged);
* in-class part: ``n_clusters`` tight clusters of ``cluster_size`` mutually
  near-duplicate demonstrations of the target class, built as
  seed = highest-similarity unused pool image to the query-batch summary, then the
  ``cluster_size - 1`` unused pool images most similar to that seed;
* distractor part: ``K - n_clusters * cluster_size`` images sampled uniformly from
  the other classes;
* candidate order: clusters in seed order (seed first, then its neighbours by
  decreasing similarity), then distractors -- the R110 convention of in-class first.

Everything else (query batches, episode counts, encoder features, predictor family
and training recipe, R-MUR code path, q values) is inherited unchanged from R110.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys

HERE = Path(__file__).resolve().parent
for _path in (HERE,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from demo_protocol import (  # noqa: E402
    BenchmarkConfig,
    DemoBatch,
    Features,
    _draw_query_indices,
    l2_normalise,
    project,
)

N_CLUSTERS = 4
CLUSTER_SIZE = 3
PROBE_SIZE = 60
KMEANS_ITERATIONS = 10
COPY_JITTER = 0.0  # >0 turns each cluster member into a near-copy of its seed
R110_REFERENCE_NEIGHBOURS = 8


@dataclass(frozen=True)
class RedundantPoolInfo:
    """Bookkeeping that makes the injected redundancy measurable."""

    cluster_id: np.ndarray  # (E, K) 0..n_clusters-1 for in-class, -1 for distractors
    seed_index: np.ndarray  # (E, n_clusters) candidate slot of each cluster seed
    within_similarity: np.ndarray  # (E, n_clusters) mean pairwise cosine inside a cluster
    between_similarity: np.ndarray  # (E,) mean pairwise cosine across clusters
    seed_member_similarity: np.ndarray  # (E, n_clusters) mean cosine(seed, member)
    r110_reference_similarity: np.ndarray  # (E,) mean pairwise cosine of the R110 top-8 nearest

    @property
    def episodes(self) -> int:
        return int(self.cluster_id.shape[0])


def _pairwise_mean_similarity(vectors: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity of an (n, d) set of unit vectors."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[0] < 2:
        return float("nan")
    gram = values @ values.T
    rows, columns = np.triu_indices(values.shape[0], k=1)
    return float(np.mean(gram[rows, columns]))


def build_redundant_pool(
    raw_train: np.ndarray,
    in_class_pool: np.ndarray,
    pool_compact: np.ndarray,
    query_compact: np.ndarray,
    other_pool: np.ndarray,
    *,
    rng: np.random.Generator,
    config: BenchmarkConfig,
    n_clusters: int = N_CLUSTERS,
    cluster_size: int = CLUSTER_SIZE,
    probe_size: int = PROBE_SIZE,
    kmeans_iterations: int = KMEANS_ITERATIONS,
) -> tuple[np.ndarray, dict]:
    """Build one redundancy-structured candidate pool.

    Returns the candidate *dataset indices* (K,) and a dictionary with the cluster
    assignment and the measured similarities.
    """

    candidate_count = config.candidate_count
    distractor_count = candidate_count - n_clusters * cluster_size
    if distractor_count < 0:
        raise ValueError("n_clusters * cluster_size exceeds the candidate count")

    similarity = pool_compact @ query_compact
    order = np.argsort(-similarity, kind="stable")
    working = order[: min(probe_size, order.size)]
    if working.size < n_clusters * cluster_size:
        raise ValueError(
            f"only {working.size} in-class pool images available, "
            f"need {n_clusters * cluster_size}"
        )

    # Balanced-relevance redundancy.  The working set is the ``probe_size``
    # most query-relevant in-class images; k-means with farthest-point
    # initialisation splits it into ``n_clusters`` visual modes, and each
    # cluster keeps its ``cluster_size`` most query-relevant members.  Every
    # cluster is therefore *useful* (all members come from the relevant head of
    # the class) and internally redundant (same k-means mode), which is what
    # makes a candidate's value depend on what is already selected: with
    # budget = n_clusters the sequential oracle can take one member per cluster,
    # while a standalone-utility ranking spends several slots on one mode.
    working_compact = pool_compact[working].astype(np.float64)
    # farthest-point initialisation, deterministic
    centroids = [int(np.argmax(similarity[working]))]
    while len(centroids) < n_clusters:
        distance = np.min(
            1.0 - working_compact @ working_compact[centroids].T, axis=1
        )
        distance[centroids] = -np.inf
        centroids.append(int(np.argmax(distance)))
    centroid_matrix = working_compact[centroids].copy()
    labels = np.zeros(working.size, dtype=np.int64)
    for _ in range(kmeans_iterations):
        assignment = np.argmax(working_compact @ centroid_matrix.T, axis=1)
        labels = assignment
        for cluster_index in range(n_clusters):
            members = np.flatnonzero(assignment == cluster_index)
            if members.size:
                mean = working_compact[members].mean(axis=0)
                norm = np.linalg.norm(mean)
                centroid_matrix[cluster_index] = mean / max(norm, 1e-12)

    used = np.zeros(pool_compact.shape[0], dtype=bool)
    clusters: list[list[int]] = []  # positions inside the in-class pool
    for cluster_index in range(n_clusters):
        members = working[labels == cluster_index]
        if members.size < cluster_size:
            # fall back to the most relevant unused images if a mode is tiny
            spare = working[~used[working]]
            members = np.concatenate([members, spare])[:cluster_size]
        ranked = members[np.argsort(-similarity[members], kind="stable")][:cluster_size]
        chosen_members = [int(item) for item in ranked if not used[int(item)]]
        if len(chosen_members) < cluster_size:
            spare = [int(item) for item in working if not used[int(item)] and int(item) not in chosen_members]
            chosen_members.extend(spare[: cluster_size - len(chosen_members)])
        used[np.asarray(chosen_members, dtype=np.int64)] = True
        clusters.append(chosen_members)

    candidate_slots: list[int] = []
    cluster_ids: list[int] = []
    seed_slots: list[int] = []
    within: list[float] = []
    seed_member: list[float] = []
    for cluster_index, members in enumerate(clusters):
        pool_members = np.asarray(members, dtype=np.int64)
        seed_slots.append(len(candidate_slots))
        candidate_slots.extend(int(item) for item in pool_members)
        cluster_ids.extend([cluster_index] * len(pool_members))
        within.append(_pairwise_mean_similarity(pool_compact[pool_members]))
        seed_member.append(
            float(np.mean(pool_compact[pool_members[1:]] @ pool_compact[pool_members[0]]))
        )
    # ``candidate_slots`` indexes the in-class pool array; distractors are drawn
    # directly as dataset indices, so the two parts are concatenated in dataset
    # index space at the end.
    distractors = rng.choice(other_pool, size=distractor_count, replace=False).astype(np.int64)
    cluster_ids.extend([-1] * distractor_count)

    between = _pairwise_mean_similarity(
        pool_compact[np.asarray([slot for members in clusters for slot in members], dtype=np.int64)]
    )
    reference = order[: min(R110_REFERENCE_NEIGHBOURS, order.size)]
    info = {
        "cluster_id": np.asarray(cluster_ids, dtype=np.int64),
        "seed_slot": np.asarray(seed_slots, dtype=np.int64),
        "within_similarity": np.asarray(within, dtype=np.float64),
        "between_similarity": float(between),
        "seed_member_similarity": np.asarray(seed_member, dtype=np.float64),
        "r110_reference_similarity": _pairwise_mean_similarity(pool_compact[reference]),
    }
    candidates = np.concatenate(
        [in_class_pool[np.asarray(candidate_slots, dtype=np.int64)], distractors]
    ).astype(np.int64)
    return candidates, info


def build_redundant_batch(
    features: Features,
    mean: np.ndarray,
    components: np.ndarray,
    *,
    query_source: list[np.ndarray],
    pool_source: list[np.ndarray],
    distractor_pool: np.ndarray,
    query_split: str,
    episodes_per_class: int,
    seed: int,
    config: BenchmarkConfig,
    n_clusters: int = N_CLUSTERS,
    cluster_size: int = CLUSTER_SIZE,
    probe_size: int = PROBE_SIZE,
    kmeans_iterations: int = KMEANS_ITERATIONS,
    copy_jitter: float = COPY_JITTER,
) -> tuple[DemoBatch, RedundantPoolInfo]:
    """R110 episode construction with the redundancy-structured pool rule.

    ``copy_jitter > 0`` replaces every non-seed cluster member by a perturbed
    copy of its seed (``feature + jitter * N(0, 1)``).  Real near-duplicates in
    these benchmarks are only ~0.85 cosine-similar and have *asymmetric*
    standalone utility (the seed is much more relevant than its neighbours), so
    they do not trap a standalone ranking; the perturbed-copy variant creates
    the symmetric redundancy the mechanism argument assumes and is reported as
    a separate environment.
    """

    rng = np.random.default_rng(seed)
    train_labels = np.asarray(features.train_labels)
    if query_split not in {"train", "test"}:
        raise ValueError("query_split must be 'train' or 'test'")
    raw_train = np.asarray(features.train_emb, dtype=np.float32)
    raw_query_source = (
        np.asarray(features.test_emb, dtype=np.float32) if query_split == "test" else raw_train
    )
    distractor_pool = np.asarray(distractor_pool, dtype=np.int64)
    needed_in_class = n_clusters * cluster_size
    query_rows, candidate_rows = [], []
    class_rows, episode_rows = [], []
    candidate_label_rows = []
    cluster_rows, seed_rows = [], []
    within_rows, between_rows, seed_member_rows, reference_rows = [], [], [], []
    for class_index in range(features.num_classes):
        available_query = np.asarray(query_source[class_index], dtype=np.int64)
        available_pool = np.asarray(pool_source[class_index], dtype=np.int64)
        in_class_pool = available_pool[train_labels[available_pool] == class_index]
        if in_class_pool.size < needed_in_class:
            raise ValueError(
                f"class {class_index} has only {in_class_pool.size} pool images, "
                f"need {needed_in_class}"
            )
        other_pool = distractor_pool[train_labels[distractor_pool] != class_index]
        distractor_count = config.candidate_count - needed_in_class
        if other_pool.size < distractor_count:
            raise ValueError(
                f"class {class_index} has only {other_pool.size} distractor images, "
                f"need {distractor_count}"
            )
        pool_compact = l2_normalise(project(raw_train[in_class_pool], mean, components))
        for episode in range(episodes_per_class):
            if query_split == "train":
                probe = _draw_query_indices(available_query, config.query_batch_size, rng)
                probe_compact = l2_normalise(
                    project(raw_query_source[probe].mean(axis=0, keepdims=True), mean, components)
                )[0]
                candidates, info = build_redundant_pool(
                    raw_train,
                    in_class_pool,
                    pool_compact,
                    probe_compact,
                    other_pool,
                    rng=rng,
                    config=config,
                    n_clusters=n_clusters,
                    cluster_size=cluster_size,
                    probe_size=probe_size,
                    kmeans_iterations=kmeans_iterations,
                )
                draw_source = np.setdiff1d(available_query, candidates, assume_unique=False)
                query = _draw_query_indices(draw_source, config.query_batch_size, rng)
            else:
                query = _draw_query_indices(available_query, config.query_batch_size, rng)
                query_compact = l2_normalise(
                    project(raw_query_source[query].mean(axis=0, keepdims=True), mean, components)
                )[0]
                candidates, info = build_redundant_pool(
                    raw_train,
                    in_class_pool,
                    pool_compact,
                    query_compact,
                    other_pool,
                    rng=rng,
                    config=config,
                    n_clusters=n_clusters,
                    cluster_size=cluster_size,
                    probe_size=probe_size,
                    kmeans_iterations=kmeans_iterations,
                )
            query_rows.append(query)
            candidate_rows.append(candidates)
            class_rows.append(class_index)
            episode_rows.append(episode)
            candidate_label_rows.append(train_labels[candidates])
            cluster_rows.append(info["cluster_id"])
            seed_rows.append(info["seed_slot"])
            within_rows.append(info["within_similarity"])
            between_rows.append(info["between_similarity"])
            seed_member_rows.append(info["seed_member_similarity"])
            reference_rows.append(info["r110_reference_similarity"])
    query_index = np.stack(query_rows, axis=0)
    candidate_index = np.stack(candidate_rows, axis=0)
    query_features = raw_query_source[query_index]
    candidate_features = raw_train[candidate_index].copy()
    if copy_jitter:
        for episode, cluster_id in enumerate(cluster_rows):
            for cluster_index in range(n_clusters):
                members = np.flatnonzero(cluster_id == cluster_index)
                if members.size < 2:
                    continue
                seed = candidate_features[episode, members[0]].copy()
                noise = rng.normal(
                    scale=float(copy_jitter), size=(members.size - 1, candidate_features.shape[-1])
                ).astype(np.float32)
                candidate_features[episode, members[1:]] = seed[None, :] + noise
    batch = DemoBatch(
        query_embedding=l2_normalise(project(query_features.mean(axis=1), mean, components)),
        candidate_embedding=l2_normalise(project(candidate_features, mean, components)),
        query_features=query_features,
        candidate_features=candidate_features,
        query_labels=np.asarray(class_rows, dtype=np.int64),
        candidate_labels=np.asarray(candidate_label_rows, dtype=np.int64),
        class_index=np.asarray(class_rows, dtype=np.int64),
        episode_index=np.asarray(episode_rows, dtype=np.int64),
    )
    cluster_id = np.stack(cluster_rows, axis=0)
    seed_index = np.stack(seed_rows, axis=0)
    # Similarity statistics are re-measured on the *batch's* candidate view, so
    # they describe exactly what the router and the predictor see (this matters
    # for the perturbed-copy variant, where the copies are created after the
    # pool is chosen).
    compact = batch.candidate_embedding.astype(np.float64)
    within_measured = np.full((batch.episodes, n_clusters), np.nan)
    seed_member_measured = np.full((batch.episodes, n_clusters), np.nan)
    between_measured = np.full(batch.episodes, np.nan)
    for episode in range(batch.episodes):
        for cluster_index in range(n_clusters):
            members = np.flatnonzero(cluster_id[episode] == cluster_index)
            if members.size >= 2:
                within_measured[episode, cluster_index] = _pairwise_mean_similarity(
                    compact[episode][members]
                )
                seed_member_measured[episode, cluster_index] = float(
                    np.mean(compact[episode][members[1:]] @ compact[episode][members[0]])
                )
        in_class = np.concatenate(
            [np.flatnonzero(cluster_id[episode] == index) for index in range(n_clusters)]
        )
        between_measured[episode] = _pairwise_mean_similarity(compact[episode][in_class])
    info = RedundantPoolInfo(
        cluster_id=cluster_id,
        seed_index=seed_index,
        within_similarity=within_measured,
        between_similarity=between_measured,
        seed_member_similarity=seed_member_measured,
        r110_reference_similarity=np.asarray(reference_rows, dtype=np.float64),
    )
    return batch, info
