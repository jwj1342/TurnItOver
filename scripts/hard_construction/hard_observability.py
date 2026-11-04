"""Normal-only target readability proxies; these do not perform part segmentation."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.hard_construction.hard_geometry import project_vertices


def target_readability(reference, part, framing, view, state, policy):
    points = project_vertices(reference['geometry'][state][part].vertices, framing, view)
    size = np.ptp(points, axis=0)
    pixels = np.asarray(Image.open(io.BytesIO(reference['frames'][state][str(view.id)])).convert('RGB'), dtype=np.int16)
    if pixels.shape != (256, 256, 3):
        raise ValueError('Normal readability requires native 256 images')
    foreground = np.max(np.abs(pixels - pixels[0, 0]), axis=2) > 16
    low, high = points.min(axis=0), points.max(axis=0)
    left, right = max(0, int(np.floor(128 + low[0])) - 1), min(256, int(np.ceil(128 + high[0])) + 2)
    top, bottom = max(0, int(np.floor(128 - high[1])) - 1), min(256, int(np.ceil(128 - low[1])) + 2)
    opened = ndimage.binary_opening(foreground[top:bottom, left:right], structure=np.ones((3, 3)))
    labels, _ = ndimage.label(opened)
    areas = np.bincount(labels.ravel())[1:]
    solid_size, area = [0, 0], 0
    if areas.size:
        index = int(np.argmax(areas)) + 1
        slices = ndimage.find_objects(labels)[index - 1]
        solid_size = [s.stop - s.start for s in slices]
        area = int(areas[index - 1])
    row = {'projected_long_edge_px': float(max(size)), 'projected_short_edge_px': float(min(size)),
           'normal_solid_long_edge_px': int(max(solid_size)), 'normal_solid_short_edge_px': int(min(solid_size)),
           'normal_solid_area_px': area,
           'proxy_scope': 'normal foreground inside projected target bounds; not semantic segmentation'}
    row['readable'] = (row['projected_long_edge_px'] >= policy['normal_target_min_long_edge_px']
        and row['projected_short_edge_px'] >= policy['normal_target_min_short_edge_px']
        and row['normal_solid_long_edge_px'] >= policy['normal_target_min_solid_long_edge_px']
        and row['normal_solid_short_edge_px'] >= policy['normal_target_min_solid_short_edge_px']
        and area >= policy['normal_target_min_solid_area_px'])
    return row
