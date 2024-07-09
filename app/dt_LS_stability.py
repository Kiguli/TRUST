import sympy as sp
from SumOfSquares import *

if __name__ == "__main__":
    x, y = sp.symbols('x y')
    n = 3
    deg = 2
    M = matrix_variable('M', [x, y], deg, n, hom=False, sym=True)
    prob = SOSProblem()
    const = prob.add_matrix_sos_constraint(prob, 'z', M - sp.eye(n), [x, y])
    prob.solve()
    sym_values = [(sym, var.value) for sym, var in prob._sym_var_map.items() if var.value is not None]
    M.subs(sym_values)
