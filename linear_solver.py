def is_diagonally_dominant(matrix):
    """
    Checks if the given matrix is diagonally dominant.
    """
    n = len(matrix)
    for i in range(n):
        diagonal = abs(matrix[i][i])
        off_diagonal_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diagonal < off_diagonal_sum:
            return False
    return True

def rearrange_to_diagonally_dominant(A, b):
    """
    Tries to rearrange rows in A (and b accordingly) to make A diagonally dominant.
    Returns the new matrices and True if successful, otherwise returns the original matrices and False.
    """
    n = len(A)
    used_rows = set()
    new_A = []
    new_b = []
    for i in range(n):
        found = False
        for j in range(n):
            if j not in used_rows:
                diagonal = abs(A[j][i])
                off_diagonal_sum = sum(abs(A[j][k]) for k in range(n) if k != i)
                if diagonal >= off_diagonal_sum:
                    new_A.append(A[j])
                    new_b.append(b[j])
                    used_rows.add(j)
                    found = True
                    break
        if not found:
            return A, b, False
    return new_A, new_b, True

def vector_difference_norm(vec1, vec2):
    """
    Returns the maximum absolute difference between two vectors.
    """
    return max(abs(vec1[i][0] - vec2[i][0]) for i in range(len(vec1)))

def print_vector(vec):
    """
    Returns a string representation of the vector with 6 decimal places.
    """
    return "[" + ", ".join(f"{v[0]:.6f}" for v in vec) + "]"

def jacobi_method(A, b, tol=1e-5, max_iterations=100):
    """
    Solves Ax = b using the Jacobi iterative method.
    """
    n = len(A)
    x = [[0.0] for _ in range(n)]
    print("Jacobi Method:\n")
    for iteration in range(1, max_iterations + 1):
        x_new = [[0.0] for _ in range(n)]
        for i in range(n):
            sum_ax = sum(A[i][j] * x[j][0] for j in range(n) if j != i)
            x_new[i][0] = (b[i][0] - sum_ax) / A[i][i]
        print(f"Iteration {iteration}: {print_vector(x_new)}")
        if vector_difference_norm(x_new, x) < tol:
            print(f"\nConverged in {iteration} iterations.")
            return x_new, True, iteration
        x = x_new
    print("The system did not converge within the maximum number of iterations.")
    return None, False, max_iterations

def gauss_seidel_method(A, b, tol=1e-5, max_iterations=100):
    """
    Solves Ax = b using the Gauss-Seidel iterative method.
    """
    n = len(A)
    x = [[0.0] for _ in range(n)]
    print("Gauss-Seidel Method:\n")
    for iteration in range(1, max_iterations + 1):
        x_new = [row[:] for row in x]
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j][0] for j in range(i))
            sum2 = sum(A[i][j] * x[j][0] for j in range(i + 1, n))
            x_new[i][0] = (b[i][0] - sum1 - sum2) / A[i][i]
        print(f"Iteration {iteration}: {print_vector(x_new)}")
        if vector_difference_norm(x_new, x) < tol:
            print(f"\nConverged in {iteration} iterations.")
            return x_new, True, iteration
        x = x_new
    print("The system did not converge within the maximum number of iterations.")
    return None, False, max_iterations

# --- Main program ---
if __name__ == "__main__":
    # Define the matrix and vector
    matrixA = [
        [4, 2, 0],
        [2, 10, 4],
        [0, 4, 5]
    ]

    vectorB = [
        [2],
        [6],
        [5]
    ]

    print("Checking for diagonal dominance...")

    A = [row[:] for row in matrixA]
    b = [row[:] for row in vectorB]

    if is_diagonally_dominant(A):
        print("The matrix is diagonally dominant.\n")
        has_diagonal_dominance = True
    else:
        print("The matrix is NOT diagonally dominant. Trying to rearrange rows...\n")
        A, b, rearranged = rearrange_to_diagonally_dominant(A, b)
        if rearranged and is_diagonally_dominant(A):
            print("After rearranging, the matrix is now diagonally dominant.\n")
            has_diagonal_dominance = True
        else:
            print("Even after rearranging, the matrix is still NOT diagonally dominant.\n")
            print("There is NO guarantee that the iterative methods will converge.")
            has_diagonal_dominance = False

    # Loop menu
    while True:
        print("Choose the method you want to run:")
        print("1 - Jacobi Method")
        print("2 - Gauss-Seidel Method")
        print("3 - Exit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            result, converged, iterations = jacobi_method(A, b)
            method_name = "Jacobi"
        elif choice == '2':
            result, converged, iterations = gauss_seidel_method(A, b)
            method_name = "Gauss-Seidel"
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid input! Please enter 1, 2, or 3.\n")
            continue

        if result is not None:
            if not has_diagonal_dominance:
                print(f"\nAlthough the matrix is not diagonally dominant, the {method_name} method converged.\nSolution: {print_vector(result)}\n")
            else:
                print(f"\n{method_name} method solution: {print_vector(result)}\n")
        else:
            if not has_diagonal_dominance:
                print(f"\nThe system did NOT converge using the {method_name} method, and the matrix is not diagonally dominant.\n")
            else:
                print(f"\nThe system did NOT converge using the {method_name} method.\n")

        print("-" * 50)
