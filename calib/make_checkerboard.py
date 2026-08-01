#!/usr/bin/env python3
"""Generate a precise, printable checkerboard PDF for stereo calibration.

Produces a vector PDF at exact millimetre dimensions (so printed squares are
true to size, as long as you print at 100% / "actual size", NOT "fit to page").

Default board: 9x6 *inner corners* == 10x7 squares, 25 mm each.  That matches
the defaults in capture_pairs.py / calibrate_stereo.py.

Two safety features baked into the sheet:
  * a printed 100.0 mm reference line — measure it after printing; if it is not
    exactly 100 mm your printer scaled the page, so measure ONE square with
    calipers and pass that real value to the calibration tools via --square.
  * corner registration ticks so you can confirm nothing was cropped.

Usage:
    python3 calib/make_checkerboard.py                 # A4 + Letter, 25 mm
    python3 calib/make_checkerboard.py --square 30     # bigger squares
    python3 calib/make_checkerboard.py --cols 9 --rows 6
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM = 1.0 / 25.4  # mm -> inches (matplotlib figure units are inches)

PAGES = {
    # name: (width_mm, height_mm)  -- landscape so a wide board fits
    "A4":     (297.0, 210.0),
    "Letter": (279.4, 215.9),
}


def draw(page_name, page_mm, cols, rows, square_mm, outdir):
    # cols/rows are INNER corners -> squares = (cols+1) x (rows+1)
    nsq_x = cols + 1
    nsq_y = rows + 1
    board_w = nsq_x * square_mm
    board_h = nsq_y * square_mm

    page_w, page_h = page_mm
    if board_w > page_w - 10 or board_h > page_h - 20:
        print(f"  ! {page_name}: board {board_w:.0f}x{board_h:.0f} mm may not fit "
              f"page {page_w:.0f}x{page_h:.0f} mm — reduce --square.")

    fig = plt.figure(figsize=(page_w * MM, page_h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    ax.set_aspect("equal")

    # Center the board horizontally; leave a footer band for text.
    ox = (page_w - board_w) / 2.0
    oy = (page_h - board_h) / 2.0 + 6

    # Checkerboard squares. Top-left square black (OpenCV-friendly convention).
    for iy in range(nsq_y):
        for ix in range(nsq_x):
            if (ix + iy) % 2 == 0:
                ax.add_patch(Rectangle(
                    (ox + ix * square_mm, oy + iy * square_mm),
                    square_mm, square_mm, facecolor="black", edgecolor="none"))

    # Thin border exactly around the board (for crop verification).
    ax.add_patch(Rectangle((ox, oy), board_w, board_h,
                           fill=False, edgecolor="black", linewidth=0.3))

    # 100 mm reference line in the footer.
    ref_y = oy - 12
    rx0 = ox
    ax.plot([rx0, rx0 + 100], [ref_y, ref_y], color="black", linewidth=1.0)
    for x in (rx0, rx0 + 100):
        ax.plot([x, x], [ref_y - 2, ref_y + 2], color="black", linewidth=1.0)
    ax.text(rx0 + 50, ref_y - 6, "100.0 mm reference — measure after printing",
            ha="center", va="top", fontsize=7)

    ax.text(page_w / 2, oy + board_h + 8,
            f"Checkerboard  {cols}x{rows} inner corners  "
            f"({nsq_x}x{nsq_y} squares)   square = {square_mm:.1f} mm",
            ha="center", va="bottom", fontsize=8)
    ax.text(page_w / 2, 4,
            "Print at 100% / ACTUAL SIZE (not 'fit to page'). "
            "Tape FLAT to a rigid board. If 100 mm line is off, measure a real "
            "square and pass --square to the calib tools.",
            ha="center", va="bottom", fontsize=6, color="0.35")

    out = os.path.join(outdir, f"checkerboard_{cols}x{rows}_{square_mm:.0f}mm_{page_name}.pdf")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Generate printable checkerboard PDF")
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square", type=float, default=25.0, help="square size (mm)")
    p.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--page", choices=list(PAGES) + ["both"], default="both")
    args = p.parse_args()

    pages = PAGES.keys() if args.page == "both" else [args.page]
    print(f"Board: {args.cols}x{args.rows} inner corners, {args.square:.1f} mm squares")
    for name in pages:
        draw(name, PAGES[name], args.cols, args.rows, args.square, args.outdir)
    print("\nPrint one, tape it FLAT to something rigid (clipboard / foam board),\n"
          "then run:  python3 calib/capture_pairs.py --square <measured_mm>")


if __name__ == "__main__":
    main()
