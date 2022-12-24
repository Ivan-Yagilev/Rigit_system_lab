from numpy import linalg, array
from json import dump
from math import exp


def fy(x):
    return 4*exp(-x)-3*exp(-1000*x)


def fz(x):
    return 3*exp(-1000*x)-2*exp(-x)


def get_identity_matrix(m):
    return [[1 if i == j else 0 for j in range(m)] for i in range(m)]


def writer(x_lst, yn_lst):
    with open("output_data.json", "w") as f:
        dump({"x_lst": x_lst, "yn_lst": yn_lst}, f)


class Rk:
    def __init__(self, A, h, y0, x0, x_fin):
        self._init_cond = (y0, x0)

        self._A = array(A)
        self._h = h
        self._yn = y0
        self._x = x0
        self._b = x_fin

        self.__E = array(get_identity_matrix(len(A)))
        self._explicit_coeff = self._get_explicit_coeff()
        self._implicit_coeff = self._get_implicit_coeff()

    def _get_explicit_coeff(self):
        A_sq = self._A.dot(self._A)
        coeff = self.__E + self._h*self._A + (0.5*self._h**2)*A_sq
        return coeff

    def _get_implicit_coeff(self):
        inverse_matrix = linalg.inv(self.__E - self._h*0.25*self._A)
        k1 = inverse_matrix.dot(self._A)
        coeff = self.__E + self._h*2*k1 + (0.75*self._h**2)*(k1.dot(k1))
        return coeff

    def explicit_method(self):
        n = int((self._b - self._x)/self._h)
        x_lst = [self._x]
        yn_lst = [self._yn]
        for _ in range(1, n+1):
            self._x += self._h
            x_lst.append(self._x)
            self._yn = self._explicit_coeff.dot(self._yn)
            yn_lst.append(list(self._yn))
        writer(x_lst, yn_lst)
        self._yn, self._x = self._init_cond

    def implicit_method(self):
        n = int((self._b - self._x)/self._h)
        x_lst = [self._x]
        yn_lst = [self._yn]
        for _ in range(1, n+1):
            self._x += self._h
            x_lst.append(self._x)
            self._yn = self._implicit_coeff.dot(self._yn)
            yn_lst.append(list(self._yn))
        writer(x_lst, yn_lst)
        self._yn, self._x = self._init_cond

    def get_eigenvalue(self):
        lst = linalg.eigvals(self._A)
        S = min(lst)/max(lst)
        return f'Собственные числа = {lst}, коэффициент жесткости системы = {S}'
