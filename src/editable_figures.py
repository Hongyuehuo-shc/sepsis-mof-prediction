"""Shared settings and export helper for editable journal figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


def configure_editable_matplotlib() -> None:
    mpl.rcParams.update({
        # Arial is installed on this workstation. Keeping text as
        # text (rather than paths) means the exported SVG/PDF remains editable
        # in Adobe Illustrator while preserving the journal font.
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "svg.fonttype": "none",  # Keep text as editable text in Adobe Illustrator.
        "pdf.fonttype": 42,      # Embed TrueType fonts rather than converting text to paths.
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def save_editable_figure(figure, destination: str | Path) -> list[Path]:
    """Save a journal figure as SVG with live, Illustrator-editable text."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.with_suffix("")
    outputs = [stem.with_suffix(".svg")]
    figure.savefig(outputs[0], format="svg", bbox_inches="tight")
    return outputs


configure_editable_matplotlib()
