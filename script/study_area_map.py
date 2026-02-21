#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
study_area_map.py

Generate a study area map of the Indonesian Maritime Continent (IMC)
using SRTM15+ Earth relief data via PyGMT, with a station marker
at the sampling location.

Outputs
-------
- ../figs/fig1.png                   : High-resolution relief map.
- ../reports/study_area_map.txt   : Map metadata and coordinate summary.

Author  : Sandy Herho <sh001@ucr.edu>
Date    : 2026/02/21
License : MIT
"""

import os
from datetime import datetime

import pygmt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION = [92, 170, -20, 20]       # [west, east, south, north] degrees
RESOLUTION = "15s"                 # SRTM15+ resolution
MARKER_LON = 116.708611           # station longitude (deg E)
MARKER_LAT = -0.973056            # station latitude  (deg N)
FIG_DIR = os.path.join("..", "figs")
REPORT_DIR = os.path.join("..", "reports")
FIG_PATH = os.path.join(FIG_DIR, "fig1.png")
REPORT_PATH = os.path.join(REPORT_DIR, "study_area_map.txt")
DPI = 400


# ---------------------------------------------------------------------------
# SRTMMapPlotter
# ---------------------------------------------------------------------------
class SRTMMapPlotter:
    """Load, plot, and save SRTM15+ Earth relief data using PyGMT.

    Parameters
    ----------
    region : list of float
        Geographical bounding box ``[west, east, south, north]``.
    resolution : str, optional
        Resolution string for ``pygmt.datasets.load_earth_relief``.
    """

    def __init__(self, region, resolution="15s"):
        self.region = region
        self.resolution = resolution
        self.grid = self._load_data()
        self.figure = pygmt.Figure()

    def _load_data(self):
        """Download / cache Earth relief for the target region."""
        return pygmt.datasets.load_earth_relief(
            resolution=self.resolution, region=self.region
        )

    def plot_map(self):
        """Render the relief grid as a colour-shaded image."""
        self.figure.grdimage(
            grid=self.grid, projection="M15c", frame="a", cmap="geo"
        )

    def plot_marker(self, x, y, style="c0.3c", fill="red"):
        """Overlay a point marker on the map.

        Parameters
        ----------
        x, y : float
            Longitude and latitude of the marker.
        style : str
            GMT marker style string.
        fill : str
            Fill colour name or hex code.
        """
        self.figure.plot(x=x, y=y, style=style, fill=fill)

    def add_colorbar(self, frame=None):
        """Attach an elevation colour bar."""
        if frame is None:
            frame = ["a2000", "x+lElevation", "y+lm"]
        self.figure.colorbar(frame=frame)

    def show(self):
        """Display the figure interactively."""
        self.figure.show()

    def save(self, filepath, dpi=400):
        """Save the figure at the specified DPI."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.figure.savefig(filepath, dpi=dpi)
        print(f"[INFO] Figure saved -> {filepath} ({dpi} DPI)")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(filepath):
    """Write map metadata and coordinate summary to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = [
        "=" * 60,
        "Study Area Map Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "Region bounding box (WESN):",
        f"  West  : {REGION[0]} deg E",
        f"  East  : {REGION[1]} deg E",
        f"  South : {REGION[2]} deg N",
        f"  North : {REGION[3]} deg N",
        "",
        f"Relief data resolution : SRTM15+ ({RESOLUTION})",
        "",
        "Station marker coordinates:",
        f"  Longitude : {MARKER_LON} deg E",
        f"  Latitude  : {MARKER_LAT} deg N",
        "",
        f"Output figure : {FIG_PATH}",
        f"Output DPI    : {DPI}",
        "=" * 60,
    ]
    with open(filepath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[INFO] Report saved -> {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Build the study-area map, save figure and report."""
    plotter = SRTMMapPlotter(region=REGION, resolution=RESOLUTION)
    plotter.plot_map()
    plotter.plot_marker(x=MARKER_LON, y=MARKER_LAT)
    plotter.add_colorbar()
    plotter.save(FIG_PATH, dpi=DPI)
    write_report(REPORT_PATH)


if __name__ == "__main__":
    main()
