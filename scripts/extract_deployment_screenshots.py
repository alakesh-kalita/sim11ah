"""
One-off utility: extract a representative static frame from each deployment
scenario gif in layout_gifs/ for use as a README screenshot gallery.
Not part of the simulation pipeline -- run manually if the scenes change.
"""
import os

from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(ROOT, "layout_gifs")
OUT_DIR = os.path.join(ROOT, "docs", "screenshots")

FRAME_FRACTION = 0.45  # pick a frame partway through, past any load-in animation


def extract_frame(gif_path, out_path):
    im = Image.open(gif_path)
    n_frames = getattr(im, "n_frames", 1)
    idx = max(0, min(n_frames - 1, int(n_frames * FRAME_FRACTION)))
    im.seek(idx)
    frame = im.convert("RGB")
    frame.save(out_path, format="PNG")
    print(f"{os.path.basename(gif_path)}: frame {idx}/{n_frames} -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.lower().endswith(".gif"):
            continue
        gif_path = os.path.join(SRC_DIR, fname)
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(OUT_DIR, out_name)
        extract_frame(gif_path, out_path)


if __name__ == "__main__":
    main()
