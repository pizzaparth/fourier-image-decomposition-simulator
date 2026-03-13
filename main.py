"""
fourier_image.py
────────────────
Fourier Epicycle Drawing – FULL IMAGE PIPELINE
===============================================
Given any image file, this script:
  1. Extracts the dominant contour with OpenCV
  2. Resamples it into an evenly-spaced complex-valued signal
  3. Computes the Discrete Fourier Transform
  4. Animates the epicycle chain and traces the reconstructed drawing

Run:
    python fourier_image.py --image path/to/image.png
    python fourier_image.py --image cat.jpg --n_terms 150 --n_frames 600

Dependencies: numpy, matplotlib, opencv-python (cv2)
Install:
    pip install numpy matplotlib opencv-python
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import cv2
except ImportError:
    sys.exit('OpenCV not found. Install with: pip install opencv-python')


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – IMAGE → CONTOURS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_extract_contours(image_path: str,
                               blur_ksize: int = 5,
                               canny_lo: int = 30,
                               canny_hi: int = 100):
    """
    Load an image, detect edges with Canny, and return all contours.

    Parameters
    ----------
    image_path : path to the source image
    blur_ksize : Gaussian blur kernel size (odd integer) to reduce noise
    canny_lo   : lower threshold for Canny hysteresis
    canny_hi   : upper threshold for Canny hysteresis

    Returns
    -------
    contours   : tuple of numpy arrays, each shaped (N, 1, 2), in pixel coords
    img_shape  : (height, width) of the processed image
    """
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f'Could not read image: {image_path}')

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur reduces sensor/compression noise that would create
    # spurious edges.  ksize must be odd and positive.
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Canny edge detector: two-threshold hysteresis
    #   • pixels above canny_hi  → definite edges
    #   • pixels below canny_lo  → discarded
    #   • pixels between         → kept only if connected to a definite edge
    edges = cv2.Canny(blurred, canny_lo, canny_hi)

    # Find all contours in the binary edge map.
    #   RETR_LIST   – retrieve all contours without hierarchy
    #   CHAIN_APPROX_NONE – store every single point (no approximation)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours, gray.shape   # shape = (height, width)


def select_contour(contours, strategy: str = 'longest') -> np.ndarray:
    """
    Choose one contour from a list.

    strategy
    --------
    'longest' : use the contour with the most points (usually the silhouette)
    'largest' : use the contour with the greatest arc length (perimeter)

    Returns
    -------
    pts : float32 array of shape (N, 2) in (x, y) pixel order
    """
    if not contours:
        sys.exit('No contours found in the image. '
                 'Try adjusting --canny_lo / --canny_hi.')

    if strategy == 'longest':
        chosen = max(contours, key=lambda c: len(c))
    else:
        chosen = max(contours, key=lambda c: cv2.arcLength(c, closed=True))

    # Reshape from (N, 1, 2) → (N, 2) and convert to float
    return chosen.reshape(-1, 2).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – SIGNAL PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def contour_to_complex(pts: np.ndarray) -> np.ndarray:
    """
    Convert (x, y) pixel coordinates → complex numbers  z = x + i*y.

    Note: image y-axis points downward.  We flip the sign of y so the
    animation matches the standard mathematical orientation (y up).
    """
    return pts[:, 0] - 1j * pts[:, 1]


def resample_uniform(z: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a complex-valued curve to n evenly-spaced points along arc length.

    The raw contour from findContours may have uneven point density
    (more points on detailed regions).  A uniform arc-length parametrisation
    is required for the DFT to treat all parts of the curve equally.

    Steps
    -----
    1. Compute cumulative arc length s[k] = Σ |z[k] - z[k-1]|
    2. Map s to [0, 1]  (normalised arc length parameter)
    3. Sample at n uniformly spaced values of the parameter via interpolation
    """
    # Segment lengths
    diffs   = np.diff(z)
    seg_len = np.abs(diffs)

    # Cumulative arc length, starting at 0
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum_s[-1]

    if total == 0:
        sys.exit('Degenerate contour (zero arc length).')

    # Normalise to [0, 1]
    cum_s /= total

    # Target parameter values
    t_uniform = np.linspace(0, 1, n, endpoint=False)

    # Interpolate real and imaginary parts separately
    z_re = np.interp(t_uniform, cum_s, z.real)
    z_im = np.interp(t_uniform, cum_s, z.imag)

    return z_re + 1j * z_im


def normalize_signal(z: np.ndarray) -> np.ndarray:
    """
    Centre the curve at the origin and scale it so its bounding box
    fits inside the unit square [-1, +1]².

    Centering removes the DC offset (which would just translate the drawing).
    Scaling makes the animation independent of the original image resolution.
    """
    z = z - z.mean()                           # centre
    scale = max(np.abs(z.real).max(),
                np.abs(z.imag).max())           # half-width of bounding box
    if scale > 0:
        z /= scale
    return z


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – DISCRETE FOURIER TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def compute_dft(signal: np.ndarray) -> list:
    """
    Decompose the complex path z[n] into Fourier coefficients (epicycles).

    Mathematical background
    -----------------------
    The DFT of N samples z[0..N-1] is:

        Z[k] = Σ_{n=0}^{N-1}  z[n] · exp(-i 2π k n / N)

    The inverse (reconstruction) is:

        z(t) = (1/N) Σ_k  Z[k] · exp(i 2π k t),   t ∈ [0, 1)

    Each term is an epicycle:
        • radius          r_k  = |Z[k]| / N
        • angular speed   ω_k  = k   (full turns per period)
        • initial phase   φ_k  = arg(Z[k])

    We unfold numpy's convention (frequencies 0, 1, …, N/2, −N/2+1, …, −1)
    and sort by descending amplitude so the most influential circles are
    drawn and animated first (cosmetically nicer convergence).

    Returns
    -------
    List of dicts with keys: freq, amp, phase, coeff
    """
    N  = len(signal)
    Z  = np.fft.fft(signal)
    fs = np.fft.fftfreq(N, d=1.0 / N).astype(int)   # integer frequencies

    epicycles = []
    for k, z in zip(fs, Z):
        c = z / N    # normalise so amplitude units match the curve units
        epicycles.append({
            'freq' : int(k),
            'amp'  : float(abs(c)),
            'phase': float(np.angle(c)),
            'coeff': c,
        })

    # Sort: largest radius first
    epicycles.sort(key=lambda e: e['amp'], reverse=True)
    return epicycles


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – EPICYCLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def epicycle_positions(epicycles: list, t: float, n_terms: int) -> list:
    """
    Walk the epicycle chain and return each arm's tip position.

    Parameters
    ----------
    epicycles : sorted list from compute_dft()
    t         : normalised time in [0, 1)
    n_terms   : truncate to this many terms

    Returns
    -------
    positions : list of n_terms+1 complex numbers
        positions[0]  = origin
        positions[-1] = pen tip (reconstructed point on the curve)
    """
    positions = [0 + 0j]
    current   = 0 + 0j
    for ep in epicycles[:n_terms]:
        # Rotate the coefficient: c_k · exp(i 2π k t)
        current += ep['coeff'] * np.exp(2j * np.pi * ep['freq'] * t)
        positions.append(current)
    return positions


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

def animate(epicycles: list, n_terms: int,
            n_frames: int = 400, title: str = 'Fourier Epicycles',
            save_path: str = None):
    """
    Animate the epicycle chain and the emerging drawing.

    Parameters
    ----------
    epicycles  : sorted list from compute_dft()
    n_terms    : number of Fourier terms (epicycles) to include
    n_frames   : total frames for one complete revolution
    title      : figure title
    save_path  : if provided, save animation to this path (.gif or .mp4)
    """
    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    BG   = '#0d0d1a'
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=12, pad=10)
    ax.tick_params(colors='#444')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')

    # ── Axes limits: sum of all radii gives worst-case extent ────────────────
    total_r = sum(ep['amp'] for ep in epicycles[:n_terms])
    pad     = total_r * 1.12
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    # ── Matplotlib artists ───────────────────────────────────────────────────
    # Translucent circles (one per epicycle)
    circles = [
        plt.Circle((0, 0), ep['amp'],
                   fill=False, color="#ffffff",
                   lw=2, alpha=0.788)
        for ep in epicycles[:n_terms]
    ]
    for c in circles:
        ax.add_patch(c)

    # Radius arms
    arms = [ax.plot([], [], color="#8cd7ff", lw=0.8, alpha=0.9)[0]
            for _ in range(n_terms)]

    # The tip connector (last arm highlighted differently)
    tip_dot, = ax.plot([], [], 'o', color="#000000", ms=3, zorder=5)

    # Accumulated drawing path
    path_line, = ax.plot([], [], color="#ff37d0", lw=1.8, zorder=4)

    # Pre-compute the *closed* ideal path for reference (optional ghost)
    t_all     = np.linspace(0, 1, n_frames, endpoint=False)
    ghost_pts = np.array([epicycle_positions(epicycles, t, n_terms)[-1] for t in t_all])
    ax.plot(ghost_pts.real, ghost_pts.imag, color='white', lw=0.4, alpha=0.08, zorder=1)

    # Accumulated tip history
    trace_x: list = []
    trace_y: list = []

    # ── Frame update ─────────────────────────────────────────────────────────
    def update(frame: int):
        t = frame / (n_frames - 1)

        positions = epicycle_positions(epicycles, t, n_terms)

        for i, (ep, circle, arm) in enumerate(
                zip(epicycles[:n_terms], circles, arms)):
            # Move circle centre to current arm base
            cx, cy = positions[i].real, positions[i].imag
            circle.center = (cx, cy)

            # Draw arm: from this circle's centre to its tip
            tip = positions[i + 1]
            arm.set_data([cx, tip.real], [cy, tip.imag])

        # Mark pen tip
        tip_z = positions[-1]
        tip_dot.set_data([tip_z.real], [tip_z.imag])

        # Extend traced path
        trace_x.append(tip_z.real)
        trace_y.append(tip_z.imag)
        path_line.set_data(trace_x, trace_y)

        return circles + arms + [tip_dot, path_line]

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames + 10,
        interval=1000 / 60,
        blit=True,
        repeat=False,
    )

    if save_path:
        print(f'[anim] Saving to {save_path} …')
        writer = (animation.FFMpegWriter(fps=90)
                  if save_path.endswith('.mp4')
                  else animation.PillowWriter(fps=90))
        ani.save(save_path, writer=writer)
        print('[anim] Saved.')

    plt.tight_layout()
    plt.show()
    return ani


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-STROKE SUPPORT  (optional – uses all contours as separate strokes)
# ══════════════════════════════════════════════════════════════════════════════

def animate_multi_stroke(all_contours, n_terms_each: int = 80,
                          n_frames: int = 400,
                          min_points: int = 50,
                          title: str = 'Multi-stroke Fourier Epicycles'):
    """
    Animate each contour as an independent stroke, colour-coded.

    Useful for images with multiple disconnected regions (eyes, mouth, etc.).
    Each stroke runs its own epicycle chain, drawn in a distinct colour.
    All strokes animate simultaneously, sharing the same time parameter t.

    Parameters
    ----------
    all_contours   : raw tuple from cv2.findContours
    n_terms_each   : Fourier terms per stroke
    n_frames       : frames for one full revolution
    min_points     : ignore contours shorter than this (noise filter)
    """
    STROKE_COLORS = [
        '#ff4b6e', '#4fc3f7', '#ffd700', '#a29bfe',
        '#55efc4', '#fd79a8', '#74b9ff', '#00cec9',
    ]

    # Filter tiny contours and prepare signals
    strokes = []
    for raw in all_contours:
        pts = raw.reshape(-1, 2).astype(np.float32)
        if len(pts) < min_points:
            continue
        z   = contour_to_complex(pts)
        z   = normalize_signal(z)
        z   = resample_uniform(z, n=512)
        eps = compute_dft(z)
        strokes.append(eps)

    if not strokes:
        print('[multi] No contours survived the min_points filter.')
        return

    print(f'[multi] Animating {len(strokes)} strokes …')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_title(title, color='white', fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    paths   = [ax.plot([], [], lw=1.4, color=STROKE_COLORS[i % len(STROKE_COLORS)])[0]
               for i in range(len(strokes))]
    traces  = [[] for _ in strokes]

    def update(frame):
        t = frame / n_frames
        artists = []
        for i, (eps, path, trace) in enumerate(
                zip(strokes, paths, traces)):
            tip = epicycle_positions(eps, t, n_terms_each)[-1]
            trace.append((tip.real, tip.imag))
            xs, ys = zip(*trace) if trace else ([], [])
            path.set_data(xs, ys)
            artists.append(path)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 / 60, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()
    return ani


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC HELPER – show extracted edges & contour before animating
# ══════════════════════════════════════════════════════════════════════════════

def show_diagnostics(image_path: str, contours, chosen_pts: np.ndarray,
                     img_shape):
    """
    Display a 2-panel figure:
      Left  – Canny edge map with all detected contours (green)
      Right – selected contour only (red) on black canvas
    """
    h, w = img_shape
    canvas_all      = np.zeros((h, w, 3), dtype=np.uint8)
    canvas_selected = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.drawContours(canvas_all, contours, -1, (0, 200, 80), 1)

    pts_int = chosen_pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(canvas_selected, [pts_int], -1, (220, 60, 60), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#111')
    for ax in axes:
        ax.set_facecolor('#111')
        ax.axis('off')

    axes[0].imshow(cv2.cvtColor(canvas_all,      cv2.COLOR_BGR2RGB))
    axes[0].set_title('All contours (green)', color='white')
    axes[1].imshow(cv2.cvtColor(canvas_selected, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Selected contour (red)', color='white')

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Fourier Epicycle Drawing from an image')

    parser.add_argument('--image',      required=True,
                        help='Path to input image (JPEG, PNG, …)')
    parser.add_argument('--n_terms',    type=int, default=100,
                        help='Number of epicycles (default: 100)')
    parser.add_argument('--n_points',   type=int, default=1024,
                        help='Resampling resolution of the contour (default: 1024)')
    parser.add_argument('--n_frames',   type=int, default=400,
                        help='Animation frames per revolution (default: 400)')
    parser.add_argument('--canny_lo',   type=int, default=30,
                        help='Canny lower threshold (default: 30)')
    parser.add_argument('--canny_hi',   type=int, default=100,
                        help='Canny upper threshold (default: 100)')
    parser.add_argument('--strategy',   choices=['longest', 'largest'],
                        default='longest',
                        help='Contour selection strategy (default: longest)')
    parser.add_argument('--multi',      action='store_true',
                        help='Animate all contours as separate strokes')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Show edge-detection diagnostic plots before animating')
    parser.add_argument('--save',       default=None,
                        help='Save animation to this file (.gif or .mp4)')

    args = parser.parse_args()

    # ── Stage 1: load image and extract contours ─────────────────────────────
    print(f'[1/4] Loading image: {args.image}')
    contours, img_shape = load_and_extract_contours(
        args.image, canny_lo=args.canny_lo, canny_hi=args.canny_hi)
    print(f'      Found {len(contours)} contour(s).')

    # ── Multi-stroke mode ────────────────────────────────────────────────────
    if args.multi:
        print('[  ] Multi-stroke mode enabled.')
        animate_multi_stroke(contours, n_terms_each=args.n_terms,
                             n_frames=args.n_frames)
        return

    # ── Single-contour mode ──────────────────────────────────────────────────
    chosen_pts = select_contour(contours, strategy=args.strategy)
    print(f'      Selected contour: {len(chosen_pts)} points '
          f'(strategy: {args.strategy})')

    if args.diagnostics:
        show_diagnostics(args.image, contours, chosen_pts, img_shape)

    # ── Stage 2: signal preparation ──────────────────────────────────────────
    print(f'[2/4] Preparing signal ({args.n_points} uniform samples) …')
    z = contour_to_complex(chosen_pts)
    z = normalize_signal(z)
    z = resample_uniform(z, n=args.n_points)

    # ── Stage 3: DFT ─────────────────────────────────────────────────────────
    print(f'[3/4] Computing DFT …')
    epicycles = compute_dft(z)
    print(f'      {len(epicycles)} coefficients. '
          f'Top-5 amplitudes: '
          f'{[round(e["amp"], 4) for e in epicycles[:5]]}')

    # ── Stage 4: animate ─────────────────────────────────────────────────────
    print(f'[4/4] Animating {args.n_terms} epicycles over {args.n_frames} frames …')
    animate(epicycles,
            n_terms=args.n_terms,
            n_frames=args.n_frames,
            title=f'{args.image}  –  {args.n_terms} epicycles',
            save_path=args.save)


if __name__ == '__main__':
    main()