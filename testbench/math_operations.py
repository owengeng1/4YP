import numpy as np

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def ele_multiply_summation(vect1, vect2):
    assert isinstance(vect1, np.ndarray), "ele_multiply_summation input vectors need to be numpy arrays."
    assert isinstance(vect2, np.ndarray), "ele_multiply_summation input vectors need to be numpy arrays."
    assert np.size(vect1) == np.size(vect2), "ele_multiply_summation input vectors need to be the same size."
    return np.sum(np.multiply(vect1, vect2))

def rms_diff_norm(m1, m2):
    assert isinstance(m1, np.ndarray), "rms_diff_norm input matrices must be numpy arrays."
    assert isinstance(m2, np.ndarray), "rms_diff_norm input matrices must be numpy arrays."
    assert np.shape(m1) == np.shape(m2), "rms_diff_norm inputs must be the same size"
    ans = np.divide(np.sum(np.square(np.subtract(m1, m2))), np.size(m1))
    return ans

def calculate_magnitude(v):
    assert isinstance(v, np.ndarray), "calculate_magnitude input vector must be numpy arrays."
    return np.sqrt(v.dot(v))

def solve_quadratic(p1, p2, p3):
    assert isinstance(p1, np.ndarray), "solve_quadratic input point p1 must be numpy arrays."
    assert isinstance(p2, np.ndarray), "solve_quadratic input point p2 must be numpy arrays."
    assert isinstance(p3, np.ndarray), "solve_quadratic input point p3 must be numpy arrays."
    assert np.size(p1) == 2, "solve_quadratic input point p1 must be of size 2."
    assert np.size(p2) == 2, "solve_quadratic input point p2 must be of size 2."
    assert np.size(p3) == 2, "solve_quadratic input point p3 must be of size 2."
    A = np.array([[p1[0]*p1[0], p1[0], 1], [p2[0]*p2[0], p2[0], 1], [p3[0]*p3[0], p3[0], 1]])
    b = np.array([[p1[1]], [p2[1]], [p3[1]]])
    return np.linalg.solve(A, b)

def solve_quadratic_2(l1, l2):
    assert isinstance(l1, np.ndarray), "solve_quadratic input point l1 must be numpy arrays."
    assert isinstance(l2, np.ndarray), "solve_quadratic input point l2 must be numpy arrays."
    assert np.size(l1) == 3, "solve_quadratic input point l1 must be of size 3."
    assert np.size(l2) == 3, "solve_quadratic input point l2 must be of size 3."
    A = np.array([[l1[0]*l1[0], l1[0], 1], [l1[1]*l1[1], l1[1], 1], [l1[2]*l1[2], l1[2], 1]])
    return np.linalg.solve(A, l2)

def find_2d_vals(a, b):
    return (-b/(2*a))


def calc_orthog_basis(arr1, arr2): #Finds the projection of vector arr1 onto vector arr2
    assert isinstance(arr1, np.ndarray), "arr1 must be a numpy array in calc_1d_projections"
    assert isinstance(arr2, np.ndarray), "arr2 must be a numpy array in calc_1d_projections"
    arr2mag = calculate_magnitude(arr2)
    scalar = np.dot(arr1, arr2)/(arr2mag*arr2mag)
    return np.subtract(arr1, np.multiply(scalar, arr2))

def calc_l2_dist(v1, v2): #Finds the L2 distance between two points
    return calculate_magnitude(np.subtract(v1, v2))