from typing import Iterable, List, Dict, Any, Callable, Iterator, Optional, Sequence, Tuple
import itertools
import math
from dataclasses import dataclass
import xarray as xr
import numpy as np

@dataclass
class SweepVar:
    name: str
    values: Sequence[Any]

    @staticmethod
    def from_range(name: str, start: float, stop: float, num: int, endpoint: bool = True) -> "SweepVar":
        if num <= 0:
            raise ValueError("num must be > 0")
        if num == 1:
            vals = [start]
        else:
            step = (stop - start) / (num - (1 if endpoint else 0))
            vals = [start + i * step for i in range(num)]
        return SweepVar(name, vals)

    @staticmethod
    def from_linspace(name: str, start: float, stop: float, num: int, endpoint: bool = True) -> "SweepVar":
        return SweepVar.from_range(name, start, stop, num, endpoint)

    @staticmethod
    def from_logspace(name: str, start_exp: float, stop_exp: float, num: int, base: float = 10.0, endpoint: bool = True) -> "SweepVar":
        # start_exp/stop_exp are exponents: values = base ** exp
        if num <= 0:
            raise ValueError("num must be > 0")
        if num == 1:
            vals = [base ** start_exp]
        else:
            step = (stop_exp - start_exp) / (num - (1 if endpoint else 0))
            vals = [base ** (start_exp + i * step) for i in range(num)]
        return SweepVar(name, vals)

    @staticmethod
    def from_list(name: str, values: Iterable[Any]) -> "SweepVar":
        return SweepVar(name, list(values))


class Sweep:
    """
    A Sweep enumerates the cartesian product of provided SweepVar's.

    Parameters:
      vars: list of SweepVar
      inner_to_outer: If True (default), the last variable in `vars` is the innermost loop
                     (i.e., standard nested loops when you write `for a in A: for b in B: ...`).
                     If False, the first variable is innermost.
    """
    def __init__(self, vars: List[SweepVar], inner_to_outer: bool = True):
        if not vars:
            raise ValueError("At least one SweepVar required")
        self.vars = vars
        self.inner_to_outer = inner_to_outer

    def _order_for_product(self) -> List[SweepVar]:
        # itertools.product takes the leftmost iterable as the outermost; decide order accordingly
        return list(self.vars) if self.inner_to_outer else list(reversed(self.vars))

    @property
    def total_points(self) -> int:
        prod = 1
        for v in self.vars:
            prod *= max(1, len(v.values))
        return prod

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ordered = self._order_for_product()
        iterables = [v.values for v in ordered]
        for combo in itertools.product(*iterables):
            # re-map combo back to the original variable order
            if self.inner_to_outer:
                values_in_original_order = combo
            else:
                values_in_original_order = tuple(reversed(combo))
            yield {v.name: val for v, val in zip(self.vars, values_in_original_order)}

    def enumerate(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Yields (index, dict) where index is 0-based lexicographic index."""
        for idx, cfg in enumerate(self):
            yield idx, cfg

    def index_to_config(self, index: int) -> Dict[str, Any]:
        """Convert a flat index (0..total_points-1) to a configuration dict (lexicographic, matching __iter__)."""
        if index < 0 or index >= self.total_points:
            raise IndexError("index out of range")
        sizes = [len(v.values) for v in self.vars]
        # compute mixed-radix digits
        digits = []
        rem = index
        for s in reversed(sizes):  # compute digits for last variable first
            digits.append(rem % s)
            rem //= s
        digits = list(reversed(digits))
        return {v.name: v.values[d] for v, d in zip(self.vars, digits)}

    def map(self, func: Callable[[Dict[str, Any]], Any], *, parallel: bool = False, max_workers: Optional[int] = None):
        """
        Apply func to every configuration. Returns a list of results.
        If parallel=True, uses concurrent.futures.ThreadPoolExecutor (IO-bound safe). For CPU-bound,
        user can swap to ProcessPoolExecutor externally.
        """
        if not parallel:
            return [func(cfg) for cfg in self]
        # lazy import to avoid overhead for non-parallel runs
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * self.total_points
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for i, cfg in enumerate(self):
                futures[ex.submit(func, cfg)] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

def sweep_to_xarray_from_func(
    sweep: Sweep,
    func,
    output_dims
) -> xr.Dataset:
    """
    Evaluate `func(**coords)` at every sweep point, return an xarray.Dataset.

    Parameters
    ----------
    sweep : Sweep
        Sweep definition.
    func : callable
        Called as func(**coords) where coords has sweep var names.
        Should return dict of {result_name: scalar_or_array}.
    output_dims : dict[str, list[tuple[str, np.ndarray]]], optional
        Mapping from result_name -> list of (dim_name, coord_values) for the inner shape.
        Example:
            {
                "X": [("time", np.linspace(0,1,4))],
                "Y": [("freq", np.linspace(1e9, 2e9, 3)), ("pol", np.array(["H","V"]))]
            }
        If not given, generic names and integer coords are generated.
    """
    coords = {v.name: v.values for v in sweep.vars}
    dims = list(coords.keys())
    shape = tuple(len(vals) for vals in coords.values())

    # Probe first call to determine shapes
    first_res = func(**{name: vals[0] for name, vals in coords.items()})
    result_meta = {}
    extra_coords_global = {}  # for storing inner dim coords once

    for name, val in first_res.items():
        val = np.asarray(val)
        extra_shape = val.shape

        if output_dims and name in output_dims:
            extra_dim_info = output_dims[name]
            if len(extra_dim_info) != len(extra_shape):
                raise ValueError(f"Output dims for '{name}' do not match shape {extra_shape}")
            extra_dim_names = [dname for dname, _ in extra_dim_info]
            for dname, dvals in extra_dim_info:
                if dname not in extra_coords_global:
                    extra_coords_global[dname] = dvals
        else:
            # Generate generic dims & coords
            extra_dim_names = [f"{name}_dim{i}" for i in range(len(extra_shape))]
            for i, dname in enumerate(extra_dim_names):
                if dname not in extra_coords_global:
                    extra_coords_global[dname] = np.arange(extra_shape[i])

        result_meta[name] = (extra_shape, extra_dim_names)

    # Preallocate arrays
    result_data = {}
    for name, (extra_shape, _) in result_meta.items():
        full_shape = shape + extra_shape
        result_data[name] = np.empty(full_shape, dtype=float)

    # Fill arrays
    for idx, cfg in enumerate(sweep):
        nd_idx = np.unravel_index(idx, shape)
        res = func(**cfg)
        for name, (extra_shape, _) in result_meta.items():
            result_data[name][nd_idx] = np.asarray(res[name])

    # Build dataset coords: sweep coords + inner coords
    full_coords = coords.copy()
    full_coords.update(extra_coords_global)

    # Build dataset
    data_vars = {}
    for name, (extra_shape, extra_dims) in result_meta.items():
        all_dims = dims + extra_dims
        data_vars[name] = (all_dims, result_data[name])

    return xr.Dataset(data_vars=data_vars, coords=full_coords)

# -----------------------
# Example usage (3 variables)
# -----------------------
if __name__ == "__main__":
    # Variable A: 3 values
    A = SweepVar.from_list("A", [0, 1, 2])

    # Variable B: linear space between 0 and 1 (5 points)
    B = SweepVar.from_linspace("B", 0.0, 1.0, 5)

    # Variable C: logspace 10^0 .. 10^3 (4 points)
    C = SweepVar.from_logspace("C", 0, 3, 4, base=10.0)

    sweep = Sweep([A, B, C], inner_to_outer=True)  # C will be innermost loop
    print("Total points:", sweep.total_points)

    # iterate
    for idx, cfg in sweep.enumerate():
        print(idx, cfg)

    # map example
    def compute(cfg):
        # example: do something with cfg
        return (cfg["A"], cfg["B"], math.log10(cfg["C"]))

    results = sweep.map(compute)
    print("Sample result (first 5):", results[:5])

    # convert index -> config
    print("Config for index 7:", sweep.index_to_config(7))
