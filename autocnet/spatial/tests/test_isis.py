"""This module has tests for the isis functions."""

# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import contextlib
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

import kalasiris as isis

from autocnet.spatial import isis as si


class TestISIS(unittest.TestCase):

    def setUp(self) -> None:
        self.resourcedir = Path("test-resources")
        self.red50img = self.resourcedir / "PSP_010502_2090_RED5_0.img"
        self.red51img = self.resourcedir / "PSP_010502_2090_RED5_1.img"

        if not all((self.red50img.exists(), self.red51img.exists())):
            self.skipTest(
                f"One or more files is missing from the "
                f"{self.resourcedir.resolve()} directory. "
                f"Tests on real files skipped."
            )

        self.cube = self.red50img.with_suffix(".TestISIS.cub")
        isis.hi2isis(self.red50img, to=self.cube)
        isis.spiceinit(self.cube)

        self.map = self.cube.with_suffix(".map.cub")
        isis.cam2map(self.cube, to=self.map)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            self.cube.unlink()
            self.map.unlink()
            Path("print.prt").unlink()

    def test_point_info(self):
        d = si.point_info(self.cube, 10, 10, "image")
        self.assertEqual(10, d["Sample"])
        self.assertEqual(10, d["Line"])
        self.assertEqual(28.537311831691, d["PlanetocentricLatitude"].value)
        self.assertEqual(274.14960455269, d["PositiveEast360Longitude"].value)

        x_sample = 20
        x_lon = 274.14948072713
        y_line = 20
        y_lat = 28.537396673529

        d2 = si.point_info(self.cube, [10, x_sample], [10, y_line], "image")
        self.assertEqual(x_sample, d2[1]["Sample"])
        self.assertEqual(y_line, d2[1]["Line"])
        self.assertEqual(y_lat, d2[1]["PlanetocentricLatitude"].value)
        self.assertEqual(x_lon, d2[1]["PositiveEast360Longitude"].value)

        self.assertEqual(d, d2[0])

        d3 = si.point_info(self.cube, x_lon, y_lat, "ground")
        self.assertEqual(20.001087366213, d3["Sample"])
        self.assertEqual(20.004109124452, d3["Line"])

        d4 = si.point_info(self.map, 10, 10, "image")
        d5 = si.point_info(self.map, [10, x_sample], [10, y_line], "image")
        self.assertEqual(d4, [d5[0]])

        d6 = si.point_info(self.map, x_lon, y_lat, "ground")[0]
        self.assertEqual(961.20490394075, d6["Sample"])
        self.assertEqual(3959.4515093358, d6["Line"])

    def test_image_to_ground(self):
        lat, lon = si.image_to_ground(self.cube, 20, 20)
        self.assertEqual(274.14948072713, lon)
        self.assertEqual(28.537396673529, lat)

        lats, lons = si.image_to_ground(
            self.map, np.array([10, 20]), np.array([10, 20])
        )
        npt.assert_allclose(np.array([274.13903475, 274.13914465]), lons)
        npt.assert_allclose(np.array([28.57550764, 28.57541113]), lats)

    def test_ground_to_image(self):
        line, sample = si.ground_to_image(
            self.cube, 274.14948072713, 28.537396673529
        )
        self.assertEqual(20.004109124452, line)
        self.assertEqual(20.001087366213, sample)

        lines, samples = si.ground_to_image(
            self.map,
            np.array([274.13903475, 274.14948072713]),
            np.array([28.57550764, 28.57541113])
        )
        npt.assert_allclose(np.array([10.49999466, 20.50009032]), lines)
        npt.assert_allclose(np.array([10.5001324, 961.03569217]), samples)
