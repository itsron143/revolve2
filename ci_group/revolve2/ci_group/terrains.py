"""Standard terrains."""

import math

import numpy as np
import numpy.typing as npt
from noise import pnoise2
from pyrr import Vector3

from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose, Color
from revolve2.simulation.scene.geometry import GeometryHeightmap, GeometryPlane
from revolve2.simulation.scene.vector2 import Vector2
from revolve2.simulators.mujoco_simulator.textures import Checker
from revolve2.simulation.scene.geometry.textures import MapType


def flat(size: Vector2 = Vector2([20.0, 20.0])) -> Terrain:
    """
    Create a flat plane terrain.

    :param size: Size of the plane.
    :returns: The created terrain.
    """
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(),
                mass=0.0,
                size=size,
            )
        ]
    )


def flat_rugged(
    size: tuple[float, float],
    ruggedness: float = 0.1,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    """
    Create a flat terrain with slight ruggedness using a heightmap.

    :param size: Size of the terrain.
    :param ruggedness: The height variation across the terrain.
    :param granularity_multiplier: Multiplier for the number of edges used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant following the crater terrain

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    heights = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.0,
    )

    # Scale the ruggedness to maintain overall flatness
    max_height = ruggedness
    heights *= ruggedness

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1,
                heights=heights,
            )
        ]
    )


def mixed_terrain(
    size: tuple[float, float],
    ruggedness: float = 0.1,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    """
    Create a mixed terrain that includes a rugged, flat, and checker texture.

    :param size: Size of the terrain.
    :param ruggedness: The height variation across the terrain.
    :param granularity_multiplier: Multiplier for the number of edges used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant following the crater terrain

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    # print("Num Edges:", num_edges)

    rugged_heights = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.0,
    )

    flat_heights = np.zeros_like(rugged_heights)

    # print("Rugged heightmap shape:", rugged_heights.shape)
    # print("Rugged heightmap min", rugged_heights.min())
    # print("Rugged heightmap max", rugged_heights.max())
    # print("Flat heightmap shape:", flat_heights.shape)

    # Scale the ruggedness to maintain overall flatness
    max_height = ruggedness
    rugged_heights *= ruggedness

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(Vector3([0, 0, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1,
                heights=flat_heights,
            ),
            GeometryHeightmap(
                pose=Pose(Vector3([2 * size[0], 0, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1,
                heights=flat_heights,
                texture=Checker(
                    primary_color=Color(170, 170, 180, 255),
                    secondary_color=Color(150, 150, 150, 255),
                    map_type=MapType.MAP2D,
                ),
            ),
            GeometryHeightmap(
                pose=Pose(Vector3([4 * size[0], 0, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1,
                heights=rugged_heights,
            ),
        ]
    )


def mixed_flat_rugged(
    size: tuple[float, float],
    ruggedness: float = 0.1,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    """
    Create a mixed terrain with half flat and half slight ruggedness using a heightmap.

    :param size: Size of the terrain.
    :param ruggedness: The height variation across the terrain.
    :param granularity_multiplier: Multiplier for the number of edges used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant following the crater terrain

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    heights = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.0,
    )

    # Downscale the ruggedness to maintain overall flatness
    heights *= ruggedness

    # Set half the terrain to be flat
    midpoint = num_edges[0] // 2
    heights[:, :midpoint] = 0

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(),
                mass=0.0,
                size=Vector3([size[0], size[1], ruggedness]),
                base_thickness=0.1,
                heights=heights,
            )
        ]
    )


def mixed_flat_rugged_one_thirds(
    size: tuple[float, float],
    ruggedness: float = 0.1,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    """
    Create a mixed terrain with half flat and half slight ruggedness using a heightmap.

    :param size: Size of the terrain.
    :param ruggedness: The height variation across the terrain.
    :param granularity_multiplier: Multiplier for the number of edges used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant following the crater terrain

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    heights = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.0,
    )

    # Downscale the ruggedness to maintain overall flatness
    heights *= ruggedness

    # Set 2/3 of the terrain to be perfectly flat
    # Determine the start of the rugged third along the width
    rugged_start = num_edges[0] * 2 // 3
    # Set the first two-thirds of the heightmap to flat
    heights[:, :rugged_start] = 0

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(),
                mass=0.0,
                size=Vector3([size[0], size[1], ruggedness]),
                base_thickness=0.1,
                heights=heights,
            )
        ]
    )


def crater(
    size: tuple[float, float],
    ruggedness: float,
    curviness: float,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    r"""
    Create a crater-like terrain with rugged floor using a heightmap.

    It will look like::

        |            |
         \_        .'
           '.,^_..'

    A combination of the rugged and bowl heightmaps.

    :param size: Size of the crater.
    :param ruggedness: How coarse the ground is.
    :param curviness: Height of the edges of the crater.
    :param granularity_multiplier: Multiplier for how many edges are used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heightmap(num_edges=num_edges)

    max_height = ruggedness + curviness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged + curviness * bowl) / (ruggedness + curviness)

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            )
        ]
    )


def rugged_heightmap(
    size: tuple[float, float],
    num_edges: tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    """
    Create a rugged terrain heightmap.

    It will look like::

        ..^.__,^._.-.

    Be aware: the maximum height of the heightmap is not actually 1.
    It is around [-1,1] but not exactly.

    :param size: Size of the heightmap.
    :param num_edges: How many edges to use for the heightmap.
    :param density: How coarse the ruggedness is.
    :returns: The created heightmap as a 2 dimensional array.
    """
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            ),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )


def bowl_heightmap(
    num_edges: tuple[int, int],
) -> npt.NDArray[np.float_]:
    r"""
    Create a terrain heightmap in the shape of a bowl.

    It will look like::

        |         |
         \       /
          '.___.'

    The height of the edges of the bowl is 1.0 and the center is 0.0.

    :param num_edges: How many edges to use for the heightmap.
    :returns: The created heightmap as a 2 dimensional array.
    """
    return np.fromfunction(
        np.vectorize(
            lambda y, x: (
                (x / num_edges[0] * 2.0 - 1.0) ** 2
                + (y / num_edges[1] * 2.0 - 1.0) ** 2
                if math.sqrt(
                    (x / num_edges[0] * 2.0 - 1.0) ** 2
                    + (y / num_edges[1] * 2.0 - 1.0) ** 2
                )
                <= 1.0
                else 0.0
            ),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )
