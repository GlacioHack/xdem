# Copyright (c) 2026 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from skgstat import MetricSpace

try:
    import dask.array as da
except Exception:  # pragma: no cover
    da = None


class RegularLogLagMetricSpace(MetricSpace):
    """
    MetricSpace subclass for isotropic log-lag Monte Carlo sampling of a regular 2D grid.

    This method deals efficiently with large datasets by supporting Dask arrays for out-of-memory subsampling, and by
    anticipating the probability of nodata occuring in pairs (with iterative top-up). To sample short to long lags
    efficiently for variography, pairs are sampled by drawing separation vectors with log-uniform magnitude and
    uniformly distributed orientation, corresponding to an isotropic Monte Carlo sampling of logarithmic spatial lags.

    References
    ----------
    Sampling method is inspired from the log-lag sampling developed in Hugonnet et al. (2022), Section V-C and
    Supplementary Section II-C.

    Few literature references exist that describe this algorithm specifically. However, it was
    conceptualized fairly early, such as in Cressie (1993):
    "In isotropic settings, all directions are equivalent, and lag distances may be grouped on a
    logarithmic scale to ensure adequate sampling across short and long ranges."

    Or in fluid mechanics and turbulence, following Monin and Yaglom (1971):
    "For isotropic turbulence, ensemble averages over separation vectors are replaced by averages over
    uniformly distributed directions and logarithmically spaced magnitudes."

    Hugonnet et al. (2022): http://dx.doi.org/10.1109/JSTARS.2022.3188922
    Cressie (1993): http://dx.doi.org/10.1002/9781119115151
    Monin and Yaglom (1971). Statistical Fluid Mechanics: Mechanics of Turbulence (Vol. I). The MIT Press.

    Inheritance with MetricSpace and SciKit-GStat
    ---------------------------------------------

    Only `.dists` is intended to be used by Variogram class and others, all other methods are internal.

    Note: Deriving the full sparse matrix .dists limits pair sample sizes to ~10e8- (around 3-5 GB in memory).
    For even larger sample sizes, future developments could stream statistics bin by bin, but this is
    currently not supported due to the Variogram class design.

    Summary of algo
    ---------------
    We want to sample a large number of point pairs from a regular 2D grid such that separation distances cover
    short and long lags efficiently (approximately log-uniform in distance). We cannot enumerate all pairs due to
    the size of the grid, so we subsample.

    The core lag sampling is to:
      1) Subsample a distance r ~ Uniform(log(min_dist), log(max_dist))  (log-uniform in r)
      2) Subsample an angle  θ ~ Uniform(0, 2π)                          (isotropic)
      3) Convert (r, θ) to integer pixel offsets (ix, iy) by rounding
      4) Choose origins and compute targets using the offsets
      5) Reject out-of-bounds pairs and (optionally) reject NaN endpoints
      6) Avoid pair duplication during sampling to circumvent costly duplicate removal

    Dask specifics
    -------------
    - We never load the full array in memory.
    - We perform one global reduction at the start to estimate overall finite (not NaN/inf/nodata) fraction (f_valid),
      used to set the oversampling factor.
    - Then, iterating until top-up of valid values, for each candidate batch we read valid (finite) values at
      sampled indices using `vindex` (out-of-memory).

    Strategies (sampling_strategy)
    ------------------------------
    - "independent":
        Each pair is generated independently (origin + offset). This is the basic method that is moderately efficient.
        For 1M pairs, we have to sample 1M + 1M points (heavy graph with Dask.vindex).

    - "anchors":
        We reuse a set of random anchor points for one endpoint of each pair. Targets are generated relative to these
        anchors, so that anchor values (and their chunks) are reused across many pairs. For instance, for 1M pairs,
        we index 1000 anchors points that each match 1000 random points, so in the end we index 1k + 1M points
        (we thus use half the sample size of "independent").

    - "chunk_anchors":
        Like "anchors" but anchors are sampled from a small set of chunks per round to reduce chunk fan-out and
        task overhead for Dask. This method seems to perform the best overall in both speed and memory (default).

    - "anchor_batched":
        Structured generation: for each anchor sample multiple distances (log-uniform) and for each distance
        sample multiple angles. This produces blocks of pairs with shared anchors and controlled lag coverage.
        There might be some room for improvement in this method... which could make it more efficient to sample less
        chunks for one vindex (mostly affects speed, as memory is always kept to a single chunk size).

    Hybrid local/global (hybrid_local_fraction)
    -------------------------------------------
    Optionally, one can require that a fraction (or all) of pairs remains within the same origin chunk.
    This is situational (mostly for short-range variograms), but can massively reduce I/O overhead (both
    endpoints are always in the same chunk). The other pairs are sampled "globally" to preserve long-range lag coverage.

    NaN handling
    ------------
    To deal with NaNs without knowing their distribution ahead (Dask array), the following steps are applied:
    1. We estimate the global finite fraction f_valid by a single reduction (counting chunk per chunk), and deduce the
       probability of a random pair containing at least 1 NaN: p_pair_valid ≈ f_valid^2. For instance, 10% of NaNs
       in the array gives us a 81% chance of selecting a valid pair at random.
    2. We oversample so that random pairs will roughly match requested samples, then filter out pairs where
       either endpoint is not finite (NaN/inf).
    3. (Optional) We iterate (top-up) sampling until target count of valid pairs is reached. This is typically not
      critical for variography (it rarely matters if sample count is slightly larger or smaller).

    Notes on scalability
    --------------------
    Building a CSR matrix with more than 1e8 samples is RAM-heavy regardless of strategy.
    We use int32 indices when possible to reduce memory footprint.
    """

    def __init__(
        self,
        array,
        dx: float,
        dy: float,
        samples: int,
        *,
        # Log-lag sampling strategies with various chunk-compatibility
        sampling_strategy: str = "chunk_anchors",
        # Deduplication
        deduplicate: str = "per_anchor",
        # Distance range
        min_dist: float | None = None,
        max_dist: float | None = None,
        # Distance (raises error if not euclidean)
        dist_metric="euclidean",
        # Random seed
        seed: int | None = None,
        # Batching / Termination
        batch_pairs: int = 2_000_000,
        max_rounds: int = 50,
        max_oversample: float = 8.0,
        # Chunk / Locality
        chunks_per_round: int = 8,
        anchors_per_round: int = 20_000,
        # Parameters for anchor_batched
        distances_per_anchor: int = 8,
        angles_per_distance: int = 8,
        # Hybrid local/global
        hybrid_local_fraction: float = 0.0,
        max_local_dist: float | None = None,
        # Dtypes to optimize memory usage
        index_dtype=np.int32,
        data_dtype=np.float64,
    ):
        """
        Parameters
        ----------
        array:
            2D array-like of shape (ny, nx). Can be a NumPy array or a Dask array.
            Values may include NaNs. For Dask arrays, value access stays lazy until `.compute()`
            is called internally for finiteness checks (small vectors only).

        dx, dy:
            Pixel spacing in coordinate units (e.g., meters). Used to convert integer pixel
            offsets (ix, iy) into physical distances.

        samples:
            Target number of valid pairs (both endpoints finite) to include in the sparse distance matrix.

        sampling_strategy:
            One of:
              - "independent"
              - "anchors"
              - "chunk_anchors"
              - "anchor_batched"
            See class docstring for details and performance trade-offs.

        deduplicate:
            One of:
              - "global"
              - "per_anchor"
              - "none"
            Removes pair duplicates either by sorting them at the end ("global") or avoiding creating them per anchor
            ("per_anchor"; only for "anchor"-type strategies). Using "none" skips the removal, which is to avoid
            for variography (scipy.sparse sums the distances of duplicates, biasing the distribution), and only
            used for performance tests.

        min_dist, max_dist:
            Distance range (in coordinate units) for log-distance sampling.
            Defaults:
              - min_dist = min(dx, dy)
              - max_dist = diagonal of the grid (sqrt(((nx-1)dx)^2 + ((ny-1)dy)^2))

        seed:
            Seed for NumPy Generator used for all random sampling (reproducible runs).

        batch_pairs:
            Number of *candidate* pairs generated per round (before NaN filtering). Larger
            batches reduce Python overhead and Dask scheduling overhead, but require more RAM
            for temporary arrays.

        max_rounds:
            Maximum number of top-up rounds to reach `samples` valid pairs. Useful when NaNs
            are clustered or when local constraints make acceptance rates smaller.

        max_oversample:
            Safety cap controlling how aggressively we oversample relative to the requested
            `samples`. Candidate batch size is capped at `samples * max_oversample` (subject
            to `batch_pairs`), preventing runaway memory use when f_valid is small.

        chunks_per_round:
            For chunk-aligned strategies ("chunk_anchors" and some hybrid modes), how many
            distinct chunks are selected per round to draw anchors/origins from. Smaller values
            improve I/O locality but reduce spatial coverage per round.

        anchors_per_round:
            For anchor-based strategies, number of anchor points drawn per round.

        distances_per_anchor, angles_per_distance:
            Only used by "anchor_batched":
              - distances_per_anchor: number of distinct radii drawn per anchor (log-uniform)
              - angles_per_distance: number of angles drawn per radius
            Total pairs per anchor ≈ distances_per_anchor * angles_per_distance.

        hybrid_local_fraction:
            Fraction of candidate pairs that are forced to remain within the same chunk as their
            origin (both endpoints in the same chunk). Values in [0, 1].
            - 0.0: no local constraint (pure global)
            - 0.7: 70% local, 30% global (often a good compromise)
            Local constraint typically increases acceptance speed and reduces Dask I/O.

        max_local_dist:
            Upper bound on lag distances used for local mode. If None, derived from chunk
            diagonal in coordinate units so that most local pairs can fit within a chunk.
            Larger values allow longer local lags but increase rejection rate for same-chunk
            constraint.

        index_dtype:
            dtype for row/col indices stored in the sparse matrix. Use int32 when N < 2^31
            to save memory.

        data_dtype:
            dtype for distances. float64 is default; float32 can halve memory for `data`.
        """

        # Faking empty coords with a shape/length so that Variogram class works
        class Coords:
            def __init__(self, fake_shape: tuple[int, int]):
                self.shape = fake_shape
                self.n = fake_shape[0]

            def __len__(self) -> int:
                return self.n
        self.coords = Coords(fake_shape=(np.prod(array.shape), 2))

        # We intentionally do not call MetricSpace.__init__ here (it expects coords).
        self.array = array
        self.dx = float(dx)
        self.dy = float(dy)
        # Grid shape and total number of pixels
        ny, nx = array.shape
        self.shape = (int(ny), int(nx))
        self.N = int(ny) * int(nx)

        # User sample size
        self.samples = int(samples)

        # Random sampling
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Field expected from MetricSpace, we support only "euclidean"
        if dist_metric != "euclidean":
            raise ValueError("Log-lag sampling requires Euclidean distances")
        self.dist_metric = dist_metric

        # Default lag bounds if not provided
        self.min_dist = float(min_dist) if min_dist is not None else float(min(self.dx, self.dy))
        self.max_dist = float(max_dist) if max_dist is not None else float(
            np.hypot((nx - 1) * self.dx, (ny - 1) * self.dy)
        )
        if not (0 < self.min_dist < self.max_dist):
            raise ValueError("Require 0 < min_dist < max_dist")

        # Strategy selection
        self.sampling_strategy = sampling_strategy
        valid_strats = {"independent", "anchors", "chunk_anchors", "anchor_batched"}
        if self.sampling_strategy not in valid_strats:
            raise ValueError(f"sampling_strategy must be one of {sorted(valid_strats)}")

        # Batch / termination controls
        self.batch_pairs = int(batch_pairs)
        self.max_rounds = int(max_rounds)
        self.max_oversample = float(max_oversample)

        # Chunk/locality controls
        self.chunks_per_round = int(chunks_per_round)
        self.anchors_per_round = int(anchors_per_round)

        # Controls for "anchor_batched"
        self.distances_per_anchor = int(distances_per_anchor)
        self.angles_per_distance = int(angles_per_distance)

        # Hybrid local/global mixing
        self.hybrid_local_fraction = float(hybrid_local_fraction)
        if not (0.0 <= self.hybrid_local_fraction <= 1.0):
            raise ValueError("hybrid_local_fraction must be in [0, 1]")

        # Infer chunk shape for Dask arrays (or a "virtual" chunk for NumPy arrays)
        # This is used for chunk-aligned strategies and same-chunk constraints
        self._chunk_shape = self._infer_chunk_shape(default=(2048, 2048))
        cy, cx = self._chunk_shape

        # Max local distance defaults to chunk diagonal, so most local offsets are feasible
        if max_local_dist is None:
            self.max_local_dist = float(np.hypot((cx - 1) * self.dx, (cy - 1) * self.dy))
        else:
            self.max_local_dist = float(max_local_dist)

        # Sparse dtypes + symmetry
        self.index_dtype = index_dtype
        self.data_dtype = data_dtype

        # Deduplication mode
        deduplicate = str(deduplicate).lower()
        if deduplicate not in {"none", "per_anchor", "global"}:
            raise ValueError("deduplicate must be one of: 'none', 'per_anchor', 'global'")
        self.deduplicate = deduplicate

        # Cache for computed sparse matrix
        self._dists = None

    def __len__(self):
        """Return number of pixels in the implicit graph."""
        return self.N

    # -------------------------
    # Deduplication helpers
    # -------------------------

    def _dedup_global(
        self, i: np.ndarray, j: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove duplicate (i, j) pairs globally, keeping the first occurrence.

        Note on why: COO->CSR sums duplicates by default when converting to sparse matrices, and for variography we
        really don't want to sum the distances as it would bias our results...
        """
        key = i.astype(np.int64, copy=False) * np.int64(self.N) + j.astype(np.int64, copy=False)
        _, first = np.unique(key, return_index=True)
        return i[first], j[first], d[first]

    def _dedup_per_anchor(
        self, i: np.ndarray, j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove duplicate targets *within each anchor row* (i fixed), keeping the first occurrence.

        This is implemented without per-pair Python loops:
        - sort by i to make anchors contiguous
        - for each anchor group, run np.unique on its target indices j
        """
        if i.size == 0:
            return i, j

        order = np.argsort(i, kind="stable")
        i_s = i[order]
        j_s = j[order]

        boundaries = np.flatnonzero(np.diff(i_s)) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, i_s.size]

        keep_idx = []
        for s, e in zip(starts, stops):
            jj = j_s[s:e]
            _, first_local = np.unique(jj, return_index=True)
            keep_idx.append(order[s:e][np.sort(first_local, kind="stable")])

        keep = np.concatenate(keep_idx) if keep_idx else np.empty(0, dtype=np.int64)
        return i[keep], j[keep]

    # -------------------------
    # Chunk helpers
    # -------------------------

    def _infer_chunk_shape(self, default=(2048, 2048)) -> tuple[int, int]:
        """
        Infer chunk shape for chunk-aligned sampling.

        - For Dask arrays: use the first chunk size along each axis (typical uniform chunking).
        - For NumPy arrays: return a "virtual" chunk size; used only for sampling locality logic.
        """
        if da is not None and isinstance(self.array, da.Array):
            return (int(self.array.chunks[0][0]), int(self.array.chunks[1][0]))
        return (int(default[0]), int(default[1]))

    def _chunk_grid(self) -> tuple[int, int]:
        """
        Number of chunks in (y, x) directions, given array shape and chunk shape.
        """
        ny, nx = self.shape
        cy, cx = self._chunk_shape
        ncy = (ny + cy - 1) // cy
        ncx = (nx + cx - 1) // cx
        return ncy, ncx

    def _chunk_bounds(self, chy: int, chx: int) -> tuple[int, int, int, int]:
        """
        Return pixel bounds [y0,y1), [x0,x1) for a given chunk index (chy, chx).
        """
        ny, nx = self.shape
        cy, cx = self._chunk_shape
        y0 = chy * cy
        x0 = chx * cx
        y1 = min(y0 + cy, ny)
        x1 = min(x0 + cx, nx)
        return y0, y1, x0, x1

    def _sample_chunks(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample k chunk indices uniformly over the chunk grid.
        """
        ncy, ncx = self._chunk_grid()
        chy = self.rng.integers(0, ncy, size=k, dtype=np.int64)
        chx = self.rng.integers(0, ncx, size=k, dtype=np.int64)
        return chy, chx

    # -------------------------
    # NaN management
    # -------------------------

    def _estimate_valid_fraction(self) -> float:
        """
        Estimate fraction of finite values in the array (one scalar reduction).

        This is the only operation that scans the entire array, even if a Dask array. It is used to set
        an oversampling factor based on the approximation: P(pair valid) = f_valid^2.
        """
        if da is not None and isinstance(self.array, da.Array):
            n_valid = da.count_nonzero(da.isfinite(self.array)).compute()
        else:
            n_valid = np.count_nonzero(np.isfinite(self.array))
        f = float(n_valid) / float(self.N)
        return max(min(f, 1.0), 1e-12)

    def _finite_mask_flat(self, flat_idx: np.ndarray) -> np.ndarray:
        """
        Return boolean mask of finiteness at the provided flat indices.

        For Dask arrays this uses vindex and computes a small vector; it does not
        load the whole array.
        """
        _, nx = self.shape
        y, x = divmod(flat_idx.astype(np.int64, copy=False), nx)
        if da is not None and isinstance(self.array, da.Array):
            m = da.isfinite(self.array).vindex[y, x]
            return np.asarray(da.compute(m)[0], dtype=bool)
        else:
            return np.isfinite(self.array[y, x])

    # -------------------------
    # Sampling primitives
    # -------------------------

    def _sample_offsets(self, m: int, *, dist_hi: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample m integer pixel offsets (ix, iy) using log-distance + random angle.
        """
        # Sample r in [min_dist, dist_hi] log-uniformly
        lo = self.min_dist
        hi = min(self.max_dist, dist_hi)
        if hi <= lo:
            hi = self.max_dist
        log_r = self.rng.uniform(np.log(lo), np.log(hi), size=m)
        r = np.exp(log_r)

        # Sample θ uniformly in [0, 2PI)
        theta = self.rng.uniform(0.0, 2.0 * np.pi, size=m)

        # Convert to pixel offsets by rounding: ix = round((r cos θ)/dx), iy = round((r sin θ)/dy)
        ix = np.rint((r * np.cos(theta)) / self.dx).astype(np.int64)
        iy = np.rint((r * np.sin(theta)) / self.dy).astype(np.int64)

        # Remove (0,0) offsets (self-pairs)
        keep = (ix != 0) | (iy != 0)
        return ix[keep], iy[keep]

    def _exact_distances(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """
        Compute exact distances between flat indices i and j, based on discrete pixel offsets.
        """
        # We recompute the integer offsets from i,j to ensure the distance corresponds to the
        # snapped pixel lag, not the continuous r used during sampling.
        _, nx = self.shape
        y1, x1 = divmod(i, nx)
        y2, x2 = divmod(j, nx)
        ix = (x2 - x1).astype(np.int64)
        iy = (y2 - y1).astype(np.int64)
        # Equivalent to np.sqrt(X**2 + Y**2) here
        return np.hypot(ix * self.dx, iy * self.dy).astype(self.data_dtype, copy=False)

    # -------------------------
    # Candidate-pair generators (per strategy)
    # -------------------------

    def _pairs_independent(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate ~m candidate pairs where each pair is independent.
        """
        # Sample offsets (ix, iy) for each candidate
        ny, nx = self.shape
        ix, iy = self._sample_offsets(m, dist_hi=self.max_dist)

        # Sample origins (x, y) uniformly across the entire grid
        x = self.rng.integers(0, nx, size=ix.size, dtype=np.int64)
        y = self.rng.integers(0, ny, size=ix.size, dtype=np.int64)

        # Apply offsets to targets and reject out-of-bounds pairs
        x2 = x + ix
        y2 = y + iy
        ok = (0 <= x2) & (x2 < nx) & (0 <= y2) & (y2 < ny)
        if not np.any(ok):
            return np.empty(0, np.int64), np.empty(0, np.int64)
        i = y[ok] * nx + x[ok]
        j = y2[ok] * nx + x2[ok]

        return i.astype(np.int64), j.astype(np.int64)

    def _sample_anchors_anywhere(self, k: int) -> np.ndarray:
        """
        Sample k anchor points uniformly anywhere in the grid (flat indices).
        """
        return self.rng.integers(0, self.N, size=k, dtype=np.int64)

    def _sample_anchors_chunk_aligned(self, k: int) -> np.ndarray:
        """
        Sample k anchors, but concentrate them into a small set of chunks.

        This is intended to improve I/O locality for Dask:
        - fewer distinct chunks touched for anchors
        - repeated use of anchors amortizes chunk reads and scheduler overhead
        """
        _, nx = self.shape
        chy, chx = self._sample_chunks(self.chunks_per_round)

        anchors = []
        per_chunk = int(np.ceil(k / self.chunks_per_round))

        for t in range(self.chunks_per_round):
            y0, y1, x0, x1 = self._chunk_bounds(int(chy[t]), int(chx[t]))
            h, w = (y1 - y0), (x1 - x0)
            if h <= 0 or w <= 0:
                continue

            m = min(per_chunk, k - sum(a.size for a in anchors) if anchors else k)
            if m <= 0:
                break

            x = self.rng.integers(x0, x1, size=m, dtype=np.int64)
            y = self.rng.integers(y0, y1, size=m, dtype=np.int64)
            anchors.append(y * nx + x)

        if not anchors:
            return np.empty(0, dtype=np.int64)

        a = np.concatenate(anchors)
        if a.size > k:
            a = a[:k]
        return a

    def _pairs_from_anchors(self, anchors: np.ndarray, m_targets: int, *, local: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate ~m_targets candidate pairs relative to a provided list of anchors.

        Mechanism:
        - Repeat anchors so total anchor instances ≈ m_targets.
        - Sample one offset per anchor instance.
        - Apply offset to anchor coordinates to get target.
        - Reject out-of-bounds pairs.
        - If local=True, additionally require that target lies in the *same chunk*
          as the anchor (best-case Dask locality).
        """
        ny, nx = self.shape
        if anchors.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)

        # Repeat anchors to reach desired number of pairs
        rep = int(np.ceil(m_targets / anchors.size))
        a = np.repeat(anchors, rep)[:m_targets]

        y, x = divmod(a, nx)

        dist_hi = self.max_local_dist if local else self.max_dist
        ix, iy = self._sample_offsets(a.size, dist_hi=dist_hi)

        # Offsets arrays can be shorter due to (0,0) removal; align sizes
        n = min(a.size, ix.size)
        a = a[:n]
        y = y[:n]
        x = x[:n]
        ix = ix[:n]
        iy = iy[:n]

        x2 = x + ix
        y2 = y + iy

        if local:
            # Same-chunk constraint: anchor chunk id must match target chunk id
            cy, cx = self._chunk_shape
            chx0 = x // cx
            chy0 = y // cy
            chx2 = x2 // cx
            chy2 = y2 // cy

            ok = (
                (0 <= x2) & (x2 < nx) & (0 <= y2) & (y2 < ny) &
                (chx0 == chx2) & (chy0 == chy2)
            )
        else:
            ok = (0 <= x2) & (x2 < nx) & (0 <= y2) & (y2 < ny)

        if not np.any(ok):
            return np.empty(0, np.int64), np.empty(0, np.int64)

        i = a[ok]
        j = (y2[ok] * nx + x2[ok]).astype(np.int64)
        i = i.astype(np.int64)
        j = j.astype(np.int64)

        # Per-anchor dedup happens here (cheapest place: before Dask finiteness checks + distance compute)
        if self.deduplicate == "per_anchor":
            i, j = self._dedup_per_anchor(i, j)

        return i, j

    def _pairs_anchor_batched(self, anchors: np.ndarray, *, local: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Structured anchor-batched sampling:
          For each anchor:
            - sample `distances_per_anchor` radii (log-uniform)
            - sample `angles_per_distance` angles per radius
          Total pairs per anchor ≈ distances_per_anchor * angles_per_distance.

        This creates many pairs sharing the same anchor, which is helpful for:
        - amortizing anchor reads
        - stabilizing lag coverage (controlled product structure)
        - reducing random scatter (depending on anchor placement strategy)

        If local=True:
        - radii are capped to max_local_dist
        - and we require both endpoints to stay within the same chunk
        """
        ny, nx = self.shape
        if anchors.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)

        dist_hi = self.max_local_dist if local else self.max_dist
        lo = self.min_dist
        hi = min(self.max_dist, dist_hi)
        if hi <= lo:
            hi = self.max_dist

        # Radii per anchor (log-uniform)
        log_r = self.rng.uniform(np.log(lo), np.log(hi), size=(anchors.size, self.distances_per_anchor))
        r = np.exp(log_r)

        # Angles per (anchor, radius)
        theta = self.rng.uniform(
            0.0, 2.0 * np.pi,
            size=(anchors.size, self.distances_per_anchor, self.angles_per_distance),
        )

        # Broadcast radii into angle dimension
        r3 = r[:, :, None]

        # Convert to integer offsets in pixel units
        ix = np.rint((r3 * np.cos(theta)) / self.dx).astype(np.int64)
        iy = np.rint((r3 * np.sin(theta)) / self.dy).astype(np.int64)

        # Flatten to 1D arrays
        ix = ix.reshape(-1)
        iy = iy.reshape(-1)
        a = np.repeat(anchors, self.distances_per_anchor * self.angles_per_distance)

        # Drop rare (0,0) offsets (self-pairs)
        keep = (ix != 0) | (iy != 0)
        ix = ix[keep]
        iy = iy[keep]
        a = a[keep]

        # Map anchors to coordinates and apply offsets
        y, x = divmod(a, nx)
        x2 = x + ix
        y2 = y + iy

        if local:
            cy, cx = self._chunk_shape
            chx0 = x // cx
            chy0 = y // cy
            chx2 = x2 // cx
            chy2 = y2 // cy
            ok = (
                (0 <= x2) & (x2 < nx) & (0 <= y2) & (y2 < ny) &
                (chx0 == chx2) & (chy0 == chy2)
            )
        else:
            ok = (0 <= x2) & (x2 < nx) & (0 <= y2) & (y2 < ny)

        if not np.any(ok):
            return np.empty(0, np.int64), np.empty(0, np.int64)

        i = a[ok]
        j = (y2[ok] * nx + x2[ok]).astype(np.int64)
        i = i.astype(np.int64)
        j = j.astype(np.int64)

        if self.deduplicate == "per_anchor":
            i, j = self._dedup_per_anchor(i, j)

        return i, j

    # -------------------------
    # Strategy dispatcher
    # -------------------------

    def _generate_candidates(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate up to m candidate pairs (i, j) BEFORE NaN filtering.

        Optionally, this function applies the `hybrid_local_fraction` by splitting the work into:
          - local_m: pairs with same-chunk constraint for endpoints
          - global_m: pairs without same-chunk constraint (full-grid)

        The exact interpretation of "local" depends on the strategy (for "independent",
        local is implemented via chunk-aligned anchors + same-chunk constraint).
        """
        local_m = int(np.round(m * self.hybrid_local_fraction))
        global_m = m - local_m

        rows = []
        cols = []

        def add_pairs(i, j):
            if i.size:
                rows.append(i)
                cols.append(j)

        # Strategy: independent pairs
        if self.sampling_strategy == "independent":
            if local_m > 0:
                anchors = self._sample_anchors_chunk_aligned(min(self.anchors_per_round, local_m))
                i, j = self._pairs_from_anchors(anchors, local_m, local=True)
                add_pairs(i, j)
            if global_m > 0:
                i, j = self._pairs_independent(global_m)
                add_pairs(i, j)

        # Strategy: reuse anchors (anywhere or chunk-aligned)
        elif self.sampling_strategy in {"anchors", "chunk_anchors"}:
            if self.sampling_strategy == "anchors":
                anchors = self._sample_anchors_anywhere(self.anchors_per_round)
            else:
                anchors = self._sample_anchors_chunk_aligned(self.anchors_per_round)

            if local_m > 0:
                i, j = self._pairs_from_anchors(anchors, local_m, local=True)
                add_pairs(i, j)
            if global_m > 0:
                i, j = self._pairs_from_anchors(anchors, global_m, local=False)
                add_pairs(i, j)

        # Strategy: anchor-batched (structured blocks)
        elif self.sampling_strategy == "anchor_batched":
            per_anchor = self.distances_per_anchor * self.angles_per_distance

            if self.hybrid_local_fraction > 0:
                n_local_anchors = int(np.ceil(local_m / per_anchor)) if local_m > 0 else 0
                n_global_anchors = int(np.ceil(global_m / per_anchor)) if global_m > 0 else 0

                if n_local_anchors > 0:
                    anchors_l = self._sample_anchors_chunk_aligned(n_local_anchors)
                    i, j = self._pairs_anchor_batched(anchors_l, local=True)
                    add_pairs(i, j)

                if n_global_anchors > 0:
                    anchors_g = self._sample_anchors_anywhere(n_global_anchors)
                    i, j = self._pairs_anchor_batched(anchors_g, local=False)
                    add_pairs(i, j)
            else:
                n_anchors_needed = max(1, int(np.ceil(m / per_anchor)))
                anchors = self._sample_anchors_anywhere(n_anchors_needed)
                i, j = self._pairs_anchor_batched(anchors, local=False)
                add_pairs(i, j)

        else:
            raise RuntimeError("Unknown sampling strategy")

        if not rows:
            return np.empty(0, np.int64), np.empty(0, np.int64)

        i = np.concatenate(rows)
        j = np.concatenate(cols)

        # Truncate to exactly m candidates to control memory and batch work
        if i.size > m:
            i = i[:m]
            j = j[:m]

        return i.astype(np.int64), j.astype(np.int64)

    # -------------------------
    # Public API: dists
    # -------------------------

    @property
    def dists(self):
        """
        Return a sparse CSR matrix of (out-of-memory if using Dask array) sampled log-lag pairwise distances.

        Construction outline
        --------------------
        1) Estimate global finite fraction f_valid (one reduction operation).
        2) Use p_pair_valid ≈ f_valid^2 to decide candidate count per round.
        3) For each round:
           a) Generate candidates (i, j)
           b) Filter by finite (not NaN) endpoints (vindex for Dask)
           c) Compute exact distances for kept pairs
           d) Append (row, col, data) arrays
        4) Build CSR matrix at the end.
        """
        if self._dists is not None:
            return self._dists

        # Estimate acceptance rate due to NaNs (global approximation)
        f_valid = self._estimate_valid_fraction()
        p_pair_valid = max(f_valid * f_valid, 1e-12)

        # Check sample cannot be larger than unique "undirected" pairs (i.e. i,j = j,i)
        n_valid = int(round(f_valid * self.N))  # or better: return n_valid from _estimate_valid_fraction
        max_unique_valid = int(n_valid * (n_valid - 1) / 2)
        if self.samples >= max_unique_valid:
            warnings.warn(f"Too large pairwise sample size {self.samples}, defaulting to maximum number of finite "
                          f"undirected pairs {max_unique_valid} for this array of size {self.shape}.",
                          category=UserWarning)
            self.samples = max_unique_valid

        # Subsampling this way for a tiny array without replacement (deduplicate "per_anchor") might lead to stalling
        # We raise a warning if at least 25% of possible pairs are asked to be sampled
        if self.samples > 0.25 * max_unique_valid and self.deduplicate == "per_anchor":
            warnings.warn(f"Pairwise sample size requested {self.samples} is more than half of the maximum number of "
                          f"finite undirected pairs {max_unique_valid}, which might result in stalling for "
                          f"de-duplication method 'per_anchor'. Consider using 'global' de-duplication instead.",
                          category=UserWarning)

        target = self.samples
        remaining = target

        rows_acc = []
        cols_acc = []
        data_acc = []

        rounds = 0
        stalled = 0
        while remaining > 0 and rounds < self.max_rounds:
            rounds += 1
            before = remaining

            # Candidate count so expected kept ~ remaining:
            #   E[kept] ~ m * p_pair_valid  =>  m ~ remaining / p_pair_valid
            m = int(np.ceil(remaining / p_pair_valid))

            # Clamp to avoid runaway memory if acceptance is poor
            m = min(m, int(np.ceil(target * self.max_oversample)))

            # Ensure progress: do at least a reasonably sized batch
            m = max(m, min(self.batch_pairs, int(np.ceil(target * self.max_oversample))))
            m = min(m, max_unique_valid)  # Don’t ever request more candidates than possible

            # 1) Generate candidate pairs (still pure index work)
            i, j = self._generate_candidates(m)
            if i.size == 0:
                continue

            # 2) Filter invalid endpoints (NaNs/infs). Only sampled indices are touched
            keep = self._finite_mask_flat(i) & self._finite_mask_flat(j)
            if not np.any(keep):
                continue

            i = i[keep]
            j = j[keep]

            # 3) Keep only what we still need
            if i.size > remaining:
                i = i[:remaining]
                j = j[:remaining]

            # 4) Compute exact distances from snapped pixel offsets
            d = self._exact_distances(i, j)

            # 5) Accumulate (using smaller dtypes where possible)
            rows_acc.append(i.astype(self.index_dtype, copy=False))
            cols_acc.append(j.astype(self.index_dtype, copy=False))
            data_acc.append(d)

            remaining -= i.size

            # 6) Add stall detection (for small array with sampling too close to their max. number of valid pairs)
            if not rows_acc or remaining == before:
                stalled += 1
            else:
                stalled = 0
            if stalled >= 5:
                warnings.warn(
                    f"Iterative batch sampling stalled and exited. Consider reducing the sample size {self.samples} "
                    f"that might be too close to the maximum number of finite pairs {max_unique_valid}, using 'global' "
                    f"de-duplication, or computing pairs directly on the entire array instead.",
                    category=UserWarning)
                break

        if not rows_acc:
            # No valid pairs collected
            self._dists = sparse.csr_matrix((self.N, self.N))
            return self._dists

        # Concatenate all accepted triplets
        r = np.concatenate(rows_acc)
        c = np.concatenate(cols_acc)
        d = np.concatenate(data_acc)

        # Global dedup happens once at the end (robust but costs an extra unique)
        if self.deduplicate == "global":
            r, c, d = self._dedup_global(r, c, d)

        # Build CSR (if no duplicate removal, their distance will be summed)
        mat = sparse.csr_matrix((d, (r, c)), shape=(self.N, self.N))

        self._dists = mat
        return self._dists



# ----------------------------
# Hash-grid helpers
# ----------------------------

@dataclass(frozen=True)
class _GridSpec:
    """Hash-grid metadata."""
    cell_size: float
    x0: float
    y0: float
    nx: int  # number of cells in x
    ny: int  # number of cells in y


def _cell_aabb(spec: _GridSpec, cx: int, cy: int) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of cell (cx, cy) in coordinate units."""
    h = spec.cell_size
    x_min = spec.x0 + cx * h
    y_min = spec.y0 + cy * h
    x_max = x_min + h
    y_max = y_min + h
    return x_min, x_max, y_min, y_max


def _min_dist2_point_aabb(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """Squared minimum distance from point to AABB (0 if point inside)."""
    if x < x_min:
        dx = x_min - x
    elif x > x_max:
        dx = x - x_max
    else:
        dx = 0.0

    if y < y_min:
        dy = y_min - y
    elif y > y_max:
        dy = y - y_max
    else:
        dy = 0.0

    return dx * dx + dy * dy


def _max_dist2_point_aabb(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """Squared maximum distance from point to any corner of AABB."""
    dx1 = x - x_min
    dx2 = x - x_max
    dy1 = y - y_min
    dy2 = y - y_max
    return max(
        dx1 * dx1 + dy1 * dy1,
        dx1 * dx1 + dy2 * dy2,
        dx2 * dx2 + dy1 * dy1,
        dx2 * dx2 + dy2 * dy2,
    )


def _build_hash_grid_numpy(coords: np.ndarray, cell_size: float) -> tuple[Dict[Tuple[int, int], np.ndarray], _GridSpec]:
    """
    Build a hash-grid mapping (cx, cy) -> np.ndarray of point indices (int32).

    Implementation notes:
    - We avoid Python per-point appends by sorting a linearized cell key.
    - The dict stores small NumPy index arrays; later we concatenate only the
      cells that pass AABB culling for the current annulus.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    x0 = float(x.min())
    y0 = float(y.min())
    h = float(cell_size)

    cx = np.floor((x - x0) / h).astype(np.int32)
    cy = np.floor((y - y0) / h).astype(np.int32)

    nx = int(cx.max()) + 1
    ny = int(cy.max()) + 1

    # Linearize (cx, cy) into a single key for sorting/grouping
    # Use stride = ny+1 to avoid collisions.
    stride = np.int64(ny + 1)
    key = cx.astype(np.int64) * stride + cy.astype(np.int64)

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]

    # Group boundaries where key changes
    boundaries = np.flatnonzero(np.diff(key_sorted)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, key_sorted.size]

    grid: Dict[Tuple[int, int], np.ndarray] = {}
    for s, e in zip(starts, stops):
        k = int(key_sorted[s])
        ccx = k // (ny + 1)
        ccy = k % (ny + 1)
        grid[(int(ccx), int(ccy))] = order[s:e].astype(np.int32, copy=False)

    spec = _GridSpec(cell_size=h, x0=x0, y0=y0, nx=nx, ny=ny)
    return grid, spec


# ----------------------------
# Hash-grid helpers
# ----------------------------

@dataclass(frozen=True)
class _GridSpec:
    """Hash-grid metadata."""
    cell_size: float
    x0: float
    y0: float
    nx: int  # number of cells in x
    ny: int  # number of cells in y


def _cell_aabb(spec: _GridSpec, cx: int, cy: int) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of cell (cx, cy) in coordinate units."""
    h = spec.cell_size
    x_min = spec.x0 + cx * h
    y_min = spec.y0 + cy * h
    x_max = x_min + h
    y_max = y_min + h
    return x_min, x_max, y_min, y_max


def _min_dist2_point_aabb(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """Squared minimum distance from point to AABB (0 if point is inside)."""
    if x < x_min:
        dx = x_min - x
    elif x > x_max:
        dx = x - x_max
    else:
        dx = 0.0

    if y < y_min:
        dy = y_min - y
    elif y > y_max:
        dy = y - y_max
    else:
        dy = 0.0

    return dx * dx + dy * dy


def _max_dist2_point_aabb(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """Squared maximum distance from point to any corner of AABB."""
    dx1 = x - x_min
    dx2 = x - x_max
    dy1 = y - y_min
    dy2 = y - y_max
    return max(
        dx1 * dx1 + dy1 * dy1,
        dx1 * dx1 + dy2 * dy2,
        dx2 * dx2 + dy1 * dy1,
        dx2 * dx2 + dy2 * dy2,
    )


def _build_hash_grid_numpy(coords: np.ndarray, cell_size: float) -> tuple[Dict[Tuple[int, int], np.ndarray], _GridSpec]:
    """
    Build a hash-grid mapping (cx, cy) -> np.ndarray of point indices (int32).

    Performance notes:
    - Avoids Python per-point list appends by sorting a linearized cell key.
    - The dict stores small NumPy index arrays, which we concatenate only for
      those cells that pass cheap AABB culling in each annulus query.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    x0 = float(x.min())
    y0 = float(y.min())
    h = float(cell_size)

    cx = np.floor((x - x0) / h).astype(np.int32)
    cy = np.floor((y - y0) / h).astype(np.int32)

    nx = int(cx.max()) + 1
    ny = int(cy.max()) + 1

    stride = np.int64(ny + 1)
    key = cx.astype(np.int64) * stride + cy.astype(np.int64)

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]

    boundaries = np.flatnonzero(np.diff(key_sorted)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, key_sorted.size]

    grid: Dict[Tuple[int, int], np.ndarray] = {}
    for s, e in zip(starts, stops):
        k = int(key_sorted[s])
        ccx = k // (ny + 1)
        ccy = k % (ny + 1)
        grid[(int(ccx), int(ccy))] = order[s:e].astype(np.int32, copy=False)

    spec = _GridSpec(cell_size=h, x0=x0, y0=y0, nx=nx, ny=ny)
    return grid, spec



# ----------------------------
# Hash-grid helpers
# ----------------------------

@dataclass(frozen=True)
class _GridSpec:
    cell_size: float
    x0: float
    y0: float
    nx: int
    ny: int


def _cell_aabb(spec: _GridSpec, cx: int, cy: int) -> tuple[float, float, float, float]:
    h = spec.cell_size
    x_min = spec.x0 + cx * h
    y_min = spec.y0 + cy * h
    x_max = x_min + h
    y_max = y_min + h
    return x_min, x_max, y_min, y_max


def _min_dist2_point_aabb(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    if x < x_min:
        dx = x_min - x
    elif x > x_max:
        dx = x - x_max
    else:
        dx = 0.0
    if y < y_min:
        dy = y_min - y
    elif y > y_max:
        dy = y - y_max
    else:
        dy = 0.0
    return dx * dx + dy * dy


def _max_dist2_point_aabb(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    dx1 = x - x_min
    dx2 = x - x_max
    dy1 = y - y_min
    dy2 = y - y_max
    return max(
        dx1 * dx1 + dy1 * dy1,
        dx1 * dx1 + dy2 * dy2,
        dx2 * dx2 + dy1 * dy1,
        dx2 * dx2 + dy2 * dy2,
    )


def _build_hash_grid_numpy(coords: np.ndarray, cell_size: float) -> tuple[Dict[Tuple[int, int], np.ndarray], _GridSpec]:
    x = coords[:, 0]
    y = coords[:, 1]
    x0 = float(x.min())
    y0 = float(y.min())
    h = float(cell_size)

    cx = np.floor((x - x0) / h).astype(np.int32)
    cy = np.floor((y - y0) / h).astype(np.int32)

    nx = int(cx.max()) + 1
    ny = int(cy.max()) + 1

    stride = np.int64(ny + 1)
    key = cx.astype(np.int64) * stride + cy.astype(np.int64)

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]

    boundaries = np.flatnonzero(np.diff(key_sorted)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, key_sorted.size]

    grid: Dict[Tuple[int, int], np.ndarray] = {}
    for s, e in zip(starts, stops):
        k = int(key_sorted[s])
        ccx = k // (ny + 1)
        ccy = k % (ny + 1)
        grid[(int(ccx), int(ccy))] = order[s:e].astype(np.int32, copy=False)

    spec = _GridSpec(cell_size=h, x0=x0, y0=y0, nx=nx, ny=ny)
    return grid, spec


# ----------------------------
# Main class
# ----------------------------

class IrregularLogLagMetricSpace(MetricSpace):
    """
    Irregular 2D coordinate sampler returning a sparse distance matrix.

    Strategies:
      - "kdtree"      : exact annulus sampling via KDTree query_ball_point(r_out) + annulus filter
      - "hashgrid"    : exact annulus sampling via hash-grid + AABB culling + annulus filter
      - "nn_logvector": approximate log-distance + random angle with vectorized KDTree NN queries
                        (no distance-bias correction; accepts that long distances may appear more often)
    """

    def __init__(
        self,
        coords: np.ndarray,
        *,
        samples: int,
        min_dist: float,
        max_dist: float,
        n_bins: int = 24,
        strategy: str = "nn_logvector",

        # Exact annulus throughput controls
        anchors_per_round: int = 50_000,
        attempts_per_anchor: int = 1,
        max_rounds: int = 50,

        # Hash-grid tuning
        cell_size: Optional[float] = None,

        # nn_logvector tuning (vectorized)
        nn_alpha: float = 0.1,
        nn_batch_size: int = 250_000,
        nn_oversample: float = 2.0,   # proposals per remaining target in each batch
        nn_max_batches: int = 200,    # safety cap for low acceptance

        seed: Optional[int] = None,
        index_dtype=np.int32,
        data_dtype=np.float32,
        symmetrize: bool = False,
    ):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be shape (N, 2)")
        if coords.shape[0] < 2:
            raise ValueError("Need at least 2 points")
        if not (0.0 < min_dist < max_dist):
            raise ValueError("Require 0 < min_dist < max_dist")
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")

        self.coords = coords
        self.N = int(coords.shape[0])

        self.samples = int(samples)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)
        self.n_bins = int(n_bins)

        self.strategy = str(strategy).lower()
        if self.strategy not in {"kdtree", "hashgrid", "nn_logvector"}:
            raise ValueError("strategy must be 'kdtree', 'hashgrid', or 'nn_logvector'")

        self.anchors_per_round = int(anchors_per_round)
        self.attempts_per_anchor = int(attempts_per_anchor)
        self.max_rounds = int(max_rounds)

        self.bin_edges = np.exp(
            np.linspace(np.log(self.min_dist), np.log(self.max_dist), self.n_bins + 1)
        )

        self.rng = np.random.default_rng(seed)
        self.index_dtype = index_dtype
        self.data_dtype = data_dtype
        self.symmetrize = bool(symmetrize)

        # nn_logvector params
        self.nn_alpha = float(nn_alpha)
        if self.nn_alpha <= 0:
            raise ValueError("nn_alpha must be > 0")
        self.nn_batch_size = int(nn_batch_size)
        if self.nn_batch_size <= 0:
            raise ValueError("nn_batch_size must be > 0")
        self.nn_oversample = float(nn_oversample)
        if self.nn_oversample <= 0:
            raise ValueError("nn_oversample must be > 0")
        self.nn_max_batches = int(nn_max_batches)
        if self.nn_max_batches <= 0:
            raise ValueError("nn_max_batches must be > 0")

        # MetricSpace-like fields
        self.dist_metric = "euclidean"

        # Acceleration caches
        self._tree: Optional[cKDTree] = None
        self._grid: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self._grid_spec: Optional[_GridSpec] = None

        # Hash-grid cell size default: max_dist/8
        self.cell_size = float(cell_size) if cell_size is not None else (self.max_dist / 8.0)
        if self.cell_size <= 0:
            raise ValueError("cell_size must be > 0")

        self._dists: Optional[sparse.csr_matrix] = None

    def __len__(self) -> int:
        return self.N

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.coords)
        return self._tree

    def _ensure_grid(self) -> None:
        if self._grid is None or self._grid_spec is None:
            self._grid, self._grid_spec = _build_hash_grid_numpy(self.coords, self.cell_size)

    # Exact annulus sampling
    ########################

    def _sample_log_bin(self) -> tuple[float, float]:
        b = int(self.rng.integers(0, self.n_bins))
        return float(self.bin_edges[b]), float(self.bin_edges[b + 1])

    def _candidates_kdtree_outer(self, i: int, r_out: float) -> np.ndarray:
        return np.asarray(self.tree.query_ball_point(self.coords[i], r_out), dtype=np.int64)

    def _candidates_hashgrid_annulus(self, i: int, r_in: float, r_out: float) -> np.ndarray:
        self._ensure_grid()
        grid = self._grid
        spec = self._grid_spec
        assert grid is not None and spec is not None

        h = spec.cell_size
        xi, yi = float(self.coords[i, 0]), float(self.coords[i, 1])
        cx0 = int(np.floor((xi - spec.x0) / h))
        cy0 = int(np.floor((yi - spec.y0) / h))

        rad = int(np.ceil(r_out / h))
        rin2 = r_in * r_in
        rout2 = r_out * r_out

        pieces = []
        for cx in range(cx0 - rad, cx0 + rad + 1):
            if cx < 0 or cx >= spec.nx:
                continue
            for cy in range(cy0 - rad, cy0 + rad + 1):
                if cy < 0 or cy >= spec.ny:
                    continue
                idx = grid.get((cx, cy))
                if idx is None:
                    continue

                x_min, x_max, y_min, y_max = _cell_aabb(spec, cx, cy)

                if _min_dist2_point_aabb(xi, yi, x_min, x_max, y_min, y_max) >= rout2:
                    continue
                if _max_dist2_point_aabb(xi, yi, x_min, x_max, y_min, y_max) < rin2:
                    continue

                pieces.append(idx)

        if not pieces:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(pieces)

    def _pick_one_in_annulus(self, i: int, r_in: float, r_out: float) -> Optional[tuple[int, float]]:
        rin2 = r_in * r_in
        rout2 = r_out * r_out

        if self.strategy == "kdtree":
            cand = self._candidates_kdtree_outer(i, r_out)
            if cand.size == 0:
                return None
            dx = self.coords[cand, 0] - self.coords[i, 0]
            dy = self.coords[cand, 1] - self.coords[i, 1]
            d2 = dx * dx + dy * dy
            ok = (d2 >= rin2) & (d2 < rout2) & (cand != i)
            if not np.any(ok):
                return None
            ring = cand[ok]
            j = int(ring[self.rng.integers(0, ring.size)])
            dxj = float(self.coords[j, 0] - self.coords[i, 0])
            dyj = float(self.coords[j, 1] - self.coords[i, 1])
            return j, float(np.hypot(dxj, dyj))

        if self.strategy == "hashgrid":
            cand = self._candidates_hashgrid_annulus(i, r_in, r_out)
            if cand.size == 0:
                return None
            cand64 = cand.astype(np.int64, copy=False)
            dx = self.coords[cand64, 0] - self.coords[i, 0]
            dy = self.coords[cand64, 1] - self.coords[i, 1]
            d2 = dx * dx + dy * dy
            ok = (d2 >= rin2) & (d2 < rout2) & (cand64 != i)
            if not np.any(ok):
                return None
            ring = cand64[ok]
            j = int(ring[self.rng.integers(0, ring.size)])
            dxj = float(self.coords[j, 0] - self.coords[i, 0])
            dyj = float(self.coords[j, 1] - self.coords[i, 1])
            return j, float(np.hypot(dxj, dyj))

        raise RuntimeError("_pick_one_in_annulus called for non-annulus strategy")

    # Vectorized nn_logvector
    #########################

    def _fill_nn_logvector_vectorized(self, rows: np.ndarray, cols: np.ndarray, data: np.ndarray, start: int) -> int:
        """
        Fill arrays using vectorized proposals + vectorized KDTree NN query.

        - Sample r log-uniform in [min_dist, max_dist]
        - Sample theta uniform in [0, 2π)
        - Proposed target p = x_i + r*[cosθ, sinθ]
        - NN snap j = NN(p)
        - Accept if dist(p, x_j) <= alpha*r and j != i
        - Store realized distance dist(x_i, x_j)

        No correction for distance bias; this is the fastest variant.
        """
        filled = start
        target = rows.size

        log_lo = np.log(self.min_dist)
        log_hi = np.log(self.max_dist)

        batches = 0
        while filled < target and batches < self.nn_max_batches:
            batches += 1
            remaining = target - filled

            m = int(min(self.nn_batch_size, np.ceil(self.nn_oversample * remaining)))
            if m <= 0:
                break

            # Anchors
            anchors = self.rng.integers(0, self.N, size=m, dtype=np.int64)

            # Radii and angles
            r = np.exp(self.rng.uniform(log_lo, log_hi, size=m)).astype(np.float64)
            theta = self.rng.uniform(0.0, 2.0 * np.pi, size=m).astype(np.float64)

            xi = self.coords[anchors, 0]
            yi = self.coords[anchors, 1]
            px = xi + r * np.cos(theta)
            py = yi + r * np.sin(theta)

            P = np.column_stack([px, py])

            # Batch NN query
            dist_p, nn_idx = self.tree.query(P, k=1)
            nn_idx = nn_idx.astype(np.int64, copy=False)

            ok = (nn_idx != anchors) & (dist_p <= (self.nn_alpha * r))
            if not np.any(ok):
                continue

            anchors_ok = anchors[ok]
            nn_ok = nn_idx[ok]

            # Realized distances between actual points
            dx = self.coords[nn_ok, 0] - self.coords[anchors_ok, 0]
            dy = self.coords[nn_ok, 1] - self.coords[anchors_ok, 1]
            dij = np.hypot(dx, dy).astype(self.data_dtype, copy=False)

            # Take as many as needed
            take = min(remaining, dij.size)
            end = filled + take
            rows[filled:end] = anchors_ok[:take].astype(rows.dtype, copy=False)
            cols[filled:end] = nn_ok[:take].astype(cols.dtype, copy=False)
            data[filled:end] = dij[:take]
            filled = end

        return filled

    # ----------------------------
    # Output
    # ----------------------------

    @property
    def dists(self) -> sparse.csr_matrix:
        if self._dists is not None:
            return self._dists

        target = self.samples
        if target <= 0:
            self._dists = sparse.csr_matrix((self.N, self.N), dtype=self.data_dtype)
            return self._dists

        rows = np.empty(target, dtype=self.index_dtype)
        cols = np.empty(target, dtype=self.index_dtype)
        data = np.empty(target, dtype=self.data_dtype)

        filled = 0

        if self.strategy == "nn_logvector":
            filled = self._fill_nn_logvector_vectorized(rows, cols, data, filled)
        else:
            rounds = 0
            while filled < target and rounds < self.max_rounds:
                rounds += 1

                if self.anchors_per_round <= self.N:
                    anchors = self.rng.choice(self.N, size=self.anchors_per_round, replace=False)
                else:
                    anchors = self.rng.integers(0, self.N, size=self.anchors_per_round, dtype=np.int64)

                for i in anchors:
                    if filled >= target:
                        break
                    for _ in range(self.attempts_per_anchor):
                        if filled >= target:
                            break
                        r_in, r_out = self._sample_log_bin()
                        res = self._pick_one_in_annulus(int(i), r_in, r_out)
                        if res is None:
                            continue
                        j, dij = res
                        rows[filled] = int(i)
                        cols[filled] = int(j)
                        data[filled] = dij
                        filled += 1

        if filled == 0:
            self._dists = sparse.csr_matrix((self.N, self.N), dtype=self.data_dtype)
            return self._dists

        rows = rows[:filled]
        cols = cols[:filled]
        data = data[:filled]

        mat = sparse.csr_matrix((data, (rows, cols)), shape=(self.N, self.N), dtype=self.data_dtype)
        if self.symmetrize:
            mat = mat + mat.T

        self._dists = mat
        return self._dists

# ----------------------------------------
# Benchmark code (move to benchmark later)
# ----------------------------------------


# import shutil
# from time import perf_counter

# def benchmark_irregular_sampling(
#     *,
#     N: int = 50_000,
#     extent: float = 100_000.0,
#     samples: int = 500_000,
#     min_dist: float = 5.0,
#     max_dist: float = 20_000.0,
#     n_bins: int = 24,
#     anchors_per_round: int = 50_000,
#     attempts_per_anchor: int = 1,
#     max_rounds: int = 50,
#     seed: int = 0,
# ) -> None:
#     """
#     Wall-clock benchmark of all three strategies on uniform random points.
#
#     Note:
#     - nn_logvector uses KDTree internally for NN queries. It is often fastest when
#       nn_alpha is not too small (e.g., 0.1–0.2) and the point cloud is reasonably dense.
#     """
#     import time
#
#     rng = np.random.default_rng(seed)
#     coords = rng.uniform(0.0, extent, size=(N, 2)).astype(np.float64)
#
#     print(f"N={N:,}, target samples={samples:,}, bins={n_bins}, max_dist={max_dist:g}")
#
#     for strat in ["kdtree", "hashgrid", "nn_logvector"]:
#         t0 = time.perf_counter()
#         ms = IrregularLogLagMetricSpace(
#             coords,
#             samples=samples,
#             min_dist=min_dist,
#             max_dist=max_dist,
#             n_bins=n_bins,
#             strategy=strat,
#             anchors_per_round=anchors_per_round,
#             attempts_per_anchor=attempts_per_anchor,
#             max_rounds=max_rounds,
#             # hashgrid tuning
#             # cell_size=max_dist/8,
#             # nn_logvector tuning
#             nn_alpha=0.15,
#             nn_oversample=2,
#             nn_batch_size=500_000,
#             seed=seed,
#             data_dtype=np.float32,
#             symmetrize=False,
#         )
#         D = ms.dists
#         t1 = time.perf_counter()
#         print(f"{strat:12s}: nnz={D.nnz:,}  time={t1 - t0:.2f}s  mean(d)={float(D.data.mean()):.2f}")
#


#
# # 1) Create a chunked Zarr on disk
# path = "/home/atom/ongoing/own/xdem/benchmark_regulargridmetricspace/bench.zarr"
# shutil.rmtree(path, ignore_errors=True)
#
# ny, nx = 20000, 20000
# chunks = (1024, 1024)
#
# # build a dask array (still synthetic, but will be written chunked+compressed)
# arr = da.random.random((ny, nx), chunks=chunks).astype(np.float32)
# arr = da.where(arr > 0.05, arr, np.nan)
#
# # write to Zarr (this is a one-time setup cost)
# t0 = perf_counter()
# arr.to_zarr(path, overwrite=True)
# print("write seconds:", perf_counter() - t0)
#
# # 2) open lazily from disk
# arr_disk = da.from_zarr(path)
#
# # 3) run sampler/MetricSpace benchmark
# for strat in ["independent", "anchors", "chunk_anchors", "anchor_batched"]:
#     for dedup in ["none", "global", "per_anchor"]:
#         ms = RegularLogLagMetricSpace(
#             arr_disk, dx=30, dy=30,
#             samples=500_000,
#             sampling_strategy=strat,
#             deduplicate=dedup,
#             hybrid_local_fraction=0,
#             batch_pairs=2_000_000,
#             chunks_per_round=8,
#             seed=0,
#             index_dtype=np.int32,
#         )
#
#         t0 = perf_counter()
#         D = ms.dists
#         print(f"build dists seconds for strategy {strat} and dedup {dedup}:", perf_counter() - t0, "nnz:", D.nnz)
