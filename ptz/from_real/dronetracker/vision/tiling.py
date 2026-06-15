"""Split a frame into overlapping tiles for tiled YOLO inference."""

__all__ = ["split_into_tiles"]


def split_into_tiles(frame, tile_size):
    """Divide ``frame`` into an overlapping grid of tiles.

    Overlap on each axis is computed so the tiles exactly span the frame
    with no remainder; the step = tile_size - overlap.

    Args:
        frame:     numpy array (H, W, C) or (H, W).
        tile_size: (tile_height, tile_width) in pixels.

    Returns:
        (tiles, coords) where:
            tiles  — list of frame sub-arrays (views, not copies).
            coords — list of (x, y) top-left pixel positions matching each tile.
    """
    h, w   = frame.shape[:2]
    th, tw = tile_size

    # Horizontal overlap so k tiles of width tw exactly cover w pixels.
    k         = (w // tw) + 1
    overlap_x = 0 if k == 1 else ((w - k * tw) // (1 - k))

    # Vertical overlap — same logic.
    k         = (h // th) + 1
    overlap_y = 0 if k == 1 else ((h - k * th) // (1 - k))

    tiles  = []
    coords = []

    for y in range(0, h - overlap_y, th - overlap_y):
        for x in range(0, w - overlap_x, tw - overlap_x):
            tile = frame[y : y + th, x : x + tw]
            if tile.size == 0:
                continue
            tiles.append(tile)
            coords.append((x, y))

    return tiles, coords
