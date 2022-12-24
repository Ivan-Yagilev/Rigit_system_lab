import methods
from json import load
import matplotlib.pyplot as plt
from matplotlib import use
use("TkAgg")


def reader():
    with open("output_data.json", "r") as f:
        d = load(f)
        return d["x_lst"], d["yn_lst"]


with open("input_data.json", "r") as f:
    d = load(f)

A = d["A"]
h = d["h"]
y0 = d["y0"]
x0 = d["x0"]
x_fin = d["x_fin"]

rk = methods.Rk(A, h, y0, x0, x_fin)

print('\n' + rk.get_eigenvalue())

rk.explicit_method()
explic_x, explic_yn = reader()
explic_y = [i[0] for i in explic_yn]
plt.plot(explic_x, explic_y, 'r.', label='explicit y')


rk.implicit_method()
implic_x, implic_yn = reader()
implic_y = [i[0] for i in implic_yn]
plt.plot(implic_x, implic_y, 'k.', label='implicit y')


y = [methods.fy(i) for i in implic_x]
z = [methods.fz(i) for i in implic_x]
plt.plot(implic_x, y, 'b--', label='y(x)')
plt.plot(implic_x, z, 'b--', label='z(x)')


plt.suptitle('Численное решение')
plt.legend(loc='upper right')
plt.grid()
plt.show()
