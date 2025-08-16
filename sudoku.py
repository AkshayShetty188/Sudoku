import streamlit as st
from typing import List, Tuple, Optional

Grid = List[List[int]]

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="centered")
st.markdown("## 🎉 Welcome Saathvi !")
st.title("🧩 Sudoku Solver")
st.write("Enter a 9×9 Sudoku (use 0 for blanks) and click **Solve**.")

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

# --------- UI ---------
tab1, tab2 = st.tabs(["Paste 9 lines", "Manual grid"])

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

st.divider()
with st.expander("How it works"):
    st.markdown(
        """
- The app uses a **depth‑first backtracking** algorithm:
  1) Find the next empty cell, 2) try digits 1–9, 3) check row/column/box validity,
  4) recurse, 5) backtrack if a contradiction occurs.  
- `is_consistent` checks that your givens don’t already break Sudoku rules.  
- The solver returns **one** solution (most standard Sudoku have a unique solution).
        """
    )
