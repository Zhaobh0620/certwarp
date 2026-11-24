"""Q1 Gauss27 geometry for bayesreg (contract C02, C05-1, C06-3; the cell corners of C05-2a, the closed27 points of
C05-2b); layers and layouts in q1.py; ordered thread parallelism of the CPU reference in parallel.py."""
from .parallel import DEFAULT_CPU_THREADS, cpu_threads, ordered_map
from .q1 import (MAX_WHOLE_CELLS, POINT_SETS, SIMPSON_1D, TILE_SIZE, GaussGeometry, TileAdjoint, closed27, corners8,
                 finish, gauss27, gauss_geometry, gauss_gradient, gauss_gradient_energy, iter_geometry, jvp, node_field,
                 scalar_field, simpson27_weights, stiffness_apply, tile_geometry, tile_interpolate, tile_jvp,
                 tile_vjp_accumulate, tile_vjp_local, tile_vjp_scatter, vjp, vjp_stream)

__all__ = ["DEFAULT_CPU_THREADS", "MAX_WHOLE_CELLS", "POINT_SETS", "SIMPSON_1D", "TILE_SIZE", "GaussGeometry",
           "TileAdjoint", "closed27", "corners8", "cpu_threads", "finish", "ordered_map", "gauss27", "gauss_geometry",
           "gauss_gradient", "gauss_gradient_energy", "iter_geometry", "jvp", "node_field", "scalar_field",
           "simpson27_weights", "stiffness_apply", "tile_geometry", "tile_interpolate", "tile_jvp",
           "tile_vjp_accumulate", "tile_vjp_local", "tile_vjp_scatter", "vjp", "vjp_stream"]
