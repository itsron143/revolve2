from dataclasses import dataclass, field

from revolve2.simulation.scene.geometry.textures import Texture, TextureReference


@dataclass(kw_only=True, frozen=True)
class Sand(Texture):
    """A sand-like texture using a image as reference"""

    reference: TextureReference = field(
        default_factory=lambda: TextureReference(
            content_type="image/png",
            file="./sand.png"
        )
    )

    repeat: tuple[int, int] = field(default=(100, 100))
