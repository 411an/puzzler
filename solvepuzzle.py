import ctypes

# Check for duplicates in the grid
def has_conflicts(grid):
    grid_size = len(grid)
    for row in range(grid_size):
        row_values = set()  # Use a set to efficiently track seen values
        for col in range(grid_size):
            value = grid[row][col]
            if value is not None:
                if value in row_values:
                    return True  # Duplicate in row
                row_values.add(value)

    for col in range(grid_size):
        col_values = set()
        for row in range(grid_size):
            value = grid[row][col]
            if value is not None:
                if value in col_values:
                    return True  # Duplicate in column
                col_values.add(value)
    return False

# The main function
def puzzle_solution(grid, key_matrix):
    if has_conflicts(grid):
        ctypes.windll.user32.MessageBoxW(0, "Input grid contains conflicts.", "Error", 0)
        print("Input grid contains conflicts.")
        return
    else:
        solution = solve_function(grid, key_matrix)
        if solution:
            print("Puzzle solved!")
            print(solution)
            return solution
        else:  # No solution found
            ctypes.windll.user32.MessageBoxW(0, "No solution exists for this puzzle.", "Error", 0)
            print("No solution exists for this puzzle.")

def is_valid(grid, row, col, num):
    grid_size = len(grid)
    for x in range(grid_size):
        if grid[row][x] == num or grid[x][col] == num:
            return False
    return True

def solve_function(grid, key_matrix):
    def solve(grid):
        grid_size = len(grid)
        for row in range(grid_size):
            for col in range(grid_size):
                if grid[row][col] is None:
                    for num in key_matrix:
                        if is_valid(grid, row, col, num):
                            grid[row][col] = num
                            result = solve(grid)
                            if result is not None:
                                return result  # Solution found
                            grid[row][col] = None  # Backtrack
                    return None  # No valid number for this cell, no solution down this path

        # If all cells are filled and no contradictions, return the solution
        return grid

    solution = solve(grid.copy())
    if solution:
        print("Puzzle solved!")
        print(solution)
    else:
        ctypes.windll.user32.MessageBoxW(0, "No solution exists for this puzzle.", "Error", 0)
        print("No solution exists for this puzzle.")
    return solution
