# Smart Document Scanner

Turn a photo of a page into a flat, clean scan — like CamScanner, but built from
classic computer-vision steps you can read and learn from. **No AI model is used.**

Built with Python, OpenCV, NumPy, Streamlit, Pillow and ReportLab.

## Features

- Upload JPG/JPEG/PNG, use the browser camera, or try the bundled sample.
- Automatic document detection with a debug view (candidate contours, chosen outline, corners, edge map).
- Perspective correction into a flat rectangle.
- Five modes: Original Color, Enhanced Color, Grayscale, Black & White, Adaptive Threshold.
- Brightness, contrast, sharpness, 90° rotation, fine straightening, reset.
- Export PNG or JPG, or collect several scans into one multi-page PDF (A4 or Original Size).
- Clear errors for invalid files, and a retry path when detection fails.

## Project layout

```
Smart-Document-Scanner/
├── app.py                 Streamlit UI (no image logic)
├── scanner.py             Detection + perspective transform
├── image_processing.py    Scan modes and adjustments
├── pdf_generator.py       Multi-page PDF export (ReportLab)
├── utils.py               Validation, loading, encoding, resizing, exceptions
├── requirements.txt       Runtime dependencies (used by Streamlit Cloud)
├── requirements-dev.txt   Adds pytest
├── pytest.ini
├── .streamlit/config.toml Theme and upload limit
├── assets/logo.png
├── samples/
│   ├── sample_document.jpg
│   └── generate_sample.py Builds a synthetic photo with known corners (also used by tests)
└── tests/                 pytest suite
```

Data flow: `app.py` → `utils` (load) → `scanner` (detect + warp) → `image_processing` (look) → `utils`/`pdf_generator` (export).

## Run on Windows 11

Install Python 3.11 or newer from python.org (tick **Add python.exe to PATH**), then in PowerShell:

```powershell
cd Smart-Document-Scanner
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in `cmd`.
The app opens at <http://localhost:8501>. The camera works on `localhost` and on HTTPS sites.

The app uses `width="stretch"` on images, which needs a recent Streamlit.
If you see an error about that argument, run `pip install -U streamlit`.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest
```

The suite covers loading and validation, invalid images, corner ordering, corner detection,
perspective transform, every scan mode, adjustments, rotation, PDF output, and a Streamlit
smoke test (`tests/test_app.py`, skipped automatically when Streamlit is missing).

## How the algorithms work

Each step below says what it does, why it is needed, the idea behind it, the OpenCV call, the
important parameters, and when it fails. The code in `scanner.py` and `image_processing.py` has the same
explanations next to the implementation.

### 1. Resize while keeping the aspect ratio (`utils.resize_keep_aspect`)
- **What / why:** detection runs on a copy 500 px tall. A phone photo has about 12 million pixels, and
  fine texture would create thousands of false edges. The corners are scaled back up afterwards and the
  *full-resolution* photo is warped, so the scan stays sharp.
- **Idea:** `scale = new_height / old_height`; both sides use the same scale so shapes do not distort.
- **OpenCV:** `cv2.resize` with `INTER_AREA` when shrinking (averages source pixels, no aliasing).
- **Fails when:** the page is tiny in the frame. At 500 px it may be only a few dozen pixels wide.

### 2. Grayscale (`cv2.cvtColor`)
- **Why:** edges are brightness changes; colour adds noise.
- **Idea:** `gray = 0.299 R + 0.587 G + 0.114 B` (the eye is most sensitive to green).
- **Fails when:** page and background have different colours but the same brightness (white paper on a light-gray desk).

### 3. Gaussian blur (`cv2.GaussianBlur`)
- **Why:** removes sensor noise and paper texture before edge detection.
- **Idea:** each pixel becomes a weighted average of its neighbours; weights follow a bell curve.
- **Parameters:** kernel size (odd; default 5). Larger = smoother but blurrier borders.

### 4. Canny edge detection (`cv2.Canny`)
- **Why:** gives a thin line along every strong brightness change, including the page border.
- **Idea:** (1) Sobel filters measure the gradient magnitude and direction; (2) *non-maximum suppression*
  thins edges to one pixel; (3) *hysteresis*: pixels above the high threshold are edges, pixels between
  low and high are kept only if connected to a strong one.
- **Parameters:** `canny_low` (50), `canny_high` (150). Lower both for low-contrast photos.
- **Then:** a morphological closing (`cv2.morphologyEx`, `MORPH_CLOSE`) bridges tiny gaps so the border becomes a closed loop.
- **Fails when:** low contrast between page and desk, a patterned desk, shadows across the border.
- **Fallbacks:** if Canny finds nothing, the detector tries Canny with automatic thresholds
  (from the median brightness), then Otsu segmentation (`cv2.threshold` + `THRESH_OTSU`), also inverted.

### 5. Contours (`cv2.findContours`, `cv2.contourArea`)
- **Why:** turns the edge picture into shapes (lists of points) we can measure.
- **Idea:** follows the boundary of each white region. We keep the 10 largest by area.
- **Parameters:** `RETR_LIST` (all contours), `CHAIN_APPROX_SIMPLE` (stores only corner points of straight runs).

### 6. Finding the four-point polygon (`cv2.approxPolyDP`)
- **Why:** a page border is a quadrilateral; a contour has hundreds of points.
- **Idea:** the Douglas–Peucker algorithm removes points that lie within `epsilon` of the line between their
  neighbours. We use `epsilon = 0.02 × perimeter`. The first big contour that becomes exactly 4 convex
  vertices, covers at least 15 % of the photo and has sensible angles wins.
- **Relaxed pass:** if no contour simplifies to 4 points (curled page, thumb over a corner), the convex hull
  is simplified with a larger epsilon.
- **Fails when:** the page is partly out of the frame, or heavily folded.

### 7. Ordering the corners (`scanner.order_points`)
- **Why:** `getPerspectiveTransform` pairs points *by list position*. A wrong order mirrors or twists the page.
- **Idea:** sort the points by angle around their centroid (clockwise on screen), then start with the point
  with the smallest `x + y` (top-left). The textbook `x + y` / `y − x` trick breaks when the page is turned about
  45°, because two corners tie; the angle sort does not.

### 8. Perspective transform (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`)
- **Why:** a tilted page is a trapezoid in the photo. We want a rectangle.
- **Idea:** a flat surface seen by a camera is related to its front view by a 3×3 *homography* `H`
  (`[x', y', w']ᵀ = H [x, y, 1]ᵀ`, output `(x'/w', y'/w')`). `H` has 8 unknowns; each point pair gives 2
  equations, so **four corner pairs solve it exactly**. `warpPerspective` then fills every output pixel by
  looking up where it came from in the photo (inverse mapping) and interpolating.
- **Parameters:** output size = the longer of each pair of opposite sides; `INTER_CUBIC` keeps text sharp;
  `BORDER_REPLICATE` avoids black edge slivers.
- **Fails when:** the corners are wrong, even by a little: the whole page skews.

### 9. Illumination normalisation (`image_processing.normalize_illumination`)
- **Why:** photos have shadows and uneven light, so the "white" paper is different shades of gray.
- **Idea:** estimate the lighting `B(x, y)` by erasing the ink (dilation, then a median blur) and compute
  `scan = photo / B × 255`. Paper becomes white everywhere; ink stays dark.
- **Fails when:** large dark areas (photos, logos, solid blocks) are mistaken for shadow and washed out.

### 10. The five modes
| Mode | Method |
|---|---|
| Original Color | Flattened photo, untouched |
| Enhanced Color | Normalise B, G, R separately, then CLAHE on the LAB lightness channel |
| Grayscale | Normalised gray |
| Black & White | Normalised gray, then one global threshold from **Otsu's method** (maximises the variance between the dark and bright groups) |
| Adaptive Threshold | `cv2.adaptiveThreshold`: each pixel is compared with the Gaussian-weighted mean of its neighbourhood minus `C` |

- **CLAHE** (`cv2.createCLAHE`): histogram equalisation in small tiles, with `clipLimit` capping the
  contrast boost so noise is not amplified.
- **Adaptive parameters:** *block size* must be odd and larger than a stroke's width but still "local"
  (too small: hollow letters; too large: behaves like a global threshold). *C*: higher = cleaner white, but faint ink vanishes.
- **Fails when:** Otsu on a page with a big dark photo (the threshold lands in the wrong place);
  adaptive threshold on large solid areas (they turn white in the middle).

### 11. Adjustments
- **Brightness / contrast:** `out = α (in − 128) + 128 + β`, done by `cv2.convertScaleAbs`. Pivoting at 128
  means contrast darkens darks and lightens lights. In the two binary modes, brightness moves the threshold instead.
- **Sharpness:** unsharp mask, `(1 + a) · image − a · blurred`. Too much creates halos.
- **Rotation:** 90° turns use `cv2.rotate` (exact). Fine angles use `cv2.warpAffine` on an enlarged canvas.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push the project:
   ```powershell
   git init
   git add .
   git commit -m "Smart Document Scanner"
   git branch -M main
   git remote add origin https://github.com/<your-user>/Smart-Document-Scanner.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, sign in with GitHub and choose **Create app**.
3. Pick the repository, branch `main` and main file `app.py`.
4. Under **Advanced settings** choose Python 3.11 (or newer) and deploy.
5. Dependencies come from `requirements.txt`. It uses **`opencv-python-headless`** on purpose:
   the regular `opencv-python` needs system graphics libraries that the Cloud servers do not have.
6. No secrets or environment variables are needed. The camera works because Community Cloud serves over HTTPS.

Notes: free apps have limited memory (around 1 GB), which is why uploads are capped at 15 MB and the PDF at 20 pages.
`.streamlit/config.toml` holds the theme and the upload limit.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No document outline was found" | Use a darker surface, show all four corners, lower *Smallest accepted document*, lower the edge thresholds, or tick *Use the whole photo*. |
| Scan is skewed | Turn on the debug view and check the red corner dots. Adjust detection, or use the whole photo. |
| Black & White loses faint text | Lower *Brightness* (a negative value darkens the result and keeps thin strokes), or use Adaptive Threshold with a lower *Speckle filter*. |
| Text is upside down | Use *Rotate left/right*. |
| `ImportError: libGL.so.1` | You installed `opencv-python` instead of `opencv-python-headless`. |

## License

Choose a license for your repository (for example MIT) and add a `LICENSE` file.
