import streamlit as st
from typing import List, Tuple, Optional

# ==== NEW: image/OCR deps ====
import numpy as np
import cv2
import pytesseract
from PIL import Image

# If Tesseract isn't on PATH, set it (Windows example):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Grid = List[List[int]]

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="centered")
st.markdown("## 🎉 Welcome Saathvi !")
st.title("🧩 Sudoku Solver")
st.write("Enter a 9×9 Sudoku (use 0 for blanks), scan from photo, and click **Solve**.")

# --------- Core Sudoku logic ---------
def find_empty_cell(grid: Grid) -> Optional[Tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None

def valid(grid: Grid, r: int, c: int, val: int) -> bool:
    # Row
    if any(grid[r][x] == val for x in range(9)):
        return False
    # Column
    if any(grid[x][c] == val for x in range(9)):
        return False
    # Box
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == val:
                return False
    return True

def solve(grid: Grid) -> bool:
    empty = find_empty_cell(grid)
    if not empty:
        return True
    r, c = empty
    for val in range(1, 10):
        if valid(grid, r, c, val):
            grid[r][c] = val
            if solve(grid):
                return True
            grid[r][c] = 0
    return False

def grid_from_lines(lines: List[str]) -> Grid:
    g: Grid = []
    for line in lines:
        row = [int(ch) for ch in line.strip() if ch.isdigit()]
        if len(row) != 9:
            raise ValueError("Each row must contain exactly 9 digits (0–9).")
        g.append(row)
    if len(g) != 9:
        raise ValueError("Provide exactly 9 rows.")
    return g

def stringify(grid: Grid) -> str:
    return "\n".join("".join(str(x) for x in row) for row in grid)

def is_consistent(grid: Grid) -> bool:
    # Check pre-filled numbers don’t already violate rules
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            grid[r][c] = 0
            ok = valid(grid, r, c, v)
            grid[r][c] = v
            if not ok:
                return False
    return True

# --------- NEW: Scan helpers ---------
def _order_corners(pts):
    """Order 4 points (x,y) as tl, tr, br, bl."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _largest_square_contour(binary_img):
    """Find the largest 4-corner contour (the Sudoku outer border)."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best = approx
    return best

def extract_grid_image(bgr):
    """Return a top-down warped grayscale image of the Sudoku grid."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contour = _largest_square_contour(th)
    if contour is None:
        raise ValueError("Could not detect a square Sudoku grid in the image.")
    corners = contour.reshape(4, 2)
    corners = _order_corners(corners)
    side = 900  # divisible by 9
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(gray, M, (side, side))
    return warped

def split_cells(grid_img):
    """Split warped grid (square) into 81 cell images."""
    side = grid_img.shape[0]
    step = side // 9
    cells = []
    for r in range(9):
        row = []
        for c in range(9):
            cell = grid_img[r*step:(r+1)*step, c*step:(c+1)*step]
            row.append(cell)
        cells.append(row)
    return cells

def _clean_cell(cell):
    """Preprocess a cell to isolate digits (remove grid lines/noise)."""
    h, w = cell.shape
    m = int(0.12 * min(h, w))  # crop to avoid grid lines
    cell = cell[m:h-m, m:w-m]
    if cell.size == 0:
        return None
    cell = cv2.GaussianBlur(cell, (3,3), 0)
    th = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(th)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    digit = cv2.bitwise_and(th, mask)
    if cv2.countNonZero(digit) < 40:  # too tiny -> treat as blank
        return None
    digit = cv2.resize(digit, (64, 64), interpolation=cv2.INTER_AREA)
    return digit

def ocr_digit(cell_img):
    """OCR a single cell (expects preprocessed 64×64 binary digit)."""
    if cell_img is None:
        return 0
    pil = Image.fromarray(cell_img)
    # Single char, digits only
    config = "--psm 10 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(pil, config=config).strip()
    if len(txt) == 1 and txt.isdigit():
        return int(txt)
    return 0

def image_to_grid(bgr) -> Grid:
    """Full pipeline: BGR image -> 9x9 integer grid (0 for blanks)."""
    warped = extract_grid_image(bgr)
    cells = split_cells(warped)
    grid: Grid = []
    for r in range(9):
        row_vals = []
        for c in range(9):
            digit_img = _clean_cell(cells[r][c])
            val = ocr_digit(digit_img)
            row_vals.append(val)
        grid.append(row_vals)
    return grid

# --------- UI ---------
tab1, tab2, tab3 = st.tabs(["Paste 9 lines", "Manual grid", "Scan photo"])

with tab1:
    sample = "006207300\n900000004\n004010500\n100456003\n040000050\n300721008\n005080200\n800000006\n003602900"
    text = st.text_area(
        "Paste 9 lines (0 for blanks):",
        value=sample,
        height=180,
        help="Exactly 9 lines, each with 9 digits."
    )
    if st.button("Solve from text", type="primary"):
        try:
            grid = grid_from_lines(text.strip().splitlines())
            if not is_consistent(grid):
                st.error("The given puzzle is inconsistent (violates Sudoku rules).")
            else:
                g = [row[:] for row in grid]
                if solve(g):
                    st.success("Solved ✅")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Initial grid")
                        st.code(stringify(grid))
                    with col2:
                        st.caption("Solution")
                        st.code(stringify(g))
                else:
                    st.warning("This puzzle has no solution (or the solver couldn’t find one).")
        except Exception as e:
            st.error(f"Input error: {e}")

with tab2:
    st.caption("Enter numbers directly (leave 0 for blanks).")
    manual_grid: Grid = [[0]*9 for _ in range(9)]
    for r in range(9):
        cols = st.columns(9, gap="small")
        for c in range(9):
            manual_grid[r][c] = cols[c].number_input(
                label=f"R{r+1}C{c+1}",
                min_value=0, max_value=9, step=1, value=0, label_visibility="collapsed"
            )
    if st.button("Solve from manual grid", type="primary"):
        if not is_consistent(manual_grid):
            st.error("The given puzzle is inconsistent (violates Sudoku rules).")
        else:
            g = [row[:] for row in manual_grid]
            if solve(g):
                st.success("Solved ✅")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Initial grid")
                    st.code(stringify(manual_grid))
                with col2:
                    st.caption("Solution")
                    st.code(stringify(g))
            else:
                st.warning("This puzzle has no solution (or the solver couldn’t find one).")

with tab3:
    st.caption("Upload a photo or screenshot of a Sudoku. I’ll detect the grid and read digits for you.")
    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg","jpeg","png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # show preview
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

        try:
            grid = image_to_grid(bgr)
            st.success("Grid detected! Please review the parsed digits below (you can edit).")

            # Editable grid for user corrections
            edited: Grid = []
            for r in range(9):
                cols = st.columns(9, gap="small")
                row_vals = []
                for c in range(9):
                    row_vals.append(
                        cols[c].number_input(
                            label=f"R{r+1}C{c+1}",
                            min_value=0, max_value=9, step=1, value=int(grid[r][c]),
                            label_visibility="collapsed"
                        )
                    )
                edited.append(row_vals)

            if st.button("Solve scanned grid", type="primary"):
                if not is_consistent(edited):
                    st.error("The scanned puzzle violates Sudoku rules. Please correct any wrong OCR digits.")
                else:
                    g = [row[:] for row in edited]
                    if solve(g):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Parsed grid")
                            st.code(stringify(edited))
                        with col2:
                            st.caption("Solution")
                            st.code(stringify(g))
                        st.success("Solved ✅")
                    else:
                        st.warning("No solution found. Please correct misread cells and try again.")

        except Exception as e:
            st.error(f"Scan failed: {e}")

st.divider()
with st.expander("How it works"):
    st.markdown(
        """
- The app uses a **depth-first backtracking** algorithm:
  1) Find the next empty cell, 2) try digits 1–9, 3) check row/column/box validity,
  4) recurse, 5) backtrack if a contradiction occurs.  
- `is_consistent` checks that your givens don’t already break Sudoku rules.  
- The scan tab detects the grid with OpenCV, splits it into 81 cells, OCRs digits with Tesseract, then lets you edit mistakes before solving.
- The solver returns **one** solution (most standard Sudoku have a unique solution).
        """
    )
