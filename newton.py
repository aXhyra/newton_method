from matplotlib import pyplot as plt
from inspect import signature
import numpy as np
import logging
from mpl_toolkits import mplot3d


def dx(func, epsilon=1e-3):
	nargs = len(signature(func).parameters)
	if nargs == 2:
		return lambda x, y: ((func(x + epsilon, y) - func(x, y)) / epsilon)
	if nargs == 1:
		return lambda x: ((func(x + epsilon) - func(x)) / epsilon)
	logging.error('Cannot work on functions with more than 2 variables')


def dy(func, epsilon=1e-3):
	return lambda x, y: ((func(x, y + epsilon) - func(x, y)) / epsilon)


def dx_dx(func):
	nargs = len(signature(func).parameters)
	if nargs == 2:
		return lambda x, y: dx(dx(func))(x, y)
	if nargs == 1:
		return lambda x: dx(dx(func))(x)
	logging.error('Cannot work on functions with more than 2 variables')


def dy_dy(func):
	return lambda x, y: dy(dy(func))(x, y)


def dx_dy(func):
	return lambda x, y: dx(dy(func))(x, y)


def dy_dx(func):
	return lambda x, y: dy(dx(func))(x, y)


def grad(func):
	return lambda x, y: np.array([dx(func)(x, y), dy(func)(x, y)])


def hess(func):
	return lambda x, y: np.array([[dx_dx(func)(x, y), dx_dy(func)(x, y)],
	                              [dy_dx(func)(x, y), dy_dy(func)(x, y)]])


def newton(func, x0, steps=2, delta=.05):
	nargs = len(signature(func).parameters)
	if nargs == 2:
		res = np.array([x0[0], x0[1]])
		gradient = grad(func)
		hessian = hess(func)

		for i in range(steps):
			gd = gradient(res[0], res[1])
			hs = hessian(res[0], res[1])
			inverse = np.linalg.inv(hs)
			prev = res
			res = prev - (np.dot(gd, inverse))
			diff = abs(func(res[0], res[1]) - func(prev[0], prev[1]))
			if diff < delta:
				break

	elif nargs == 1:
		res = x0
		df_dx = dx(func)
		df_dxdx = dx_dx(func)

		for i in range(steps):
			prev = res
			res = prev - df_dx(prev) / df_dxdx(prev)
			diff = abs(prev - res)
			if diff < delta:
				break

	return res


def plot(func, start=-10, end=10, bins=100, savefig=False, plotfunc=True, plotgrad=True, filename=''):
	nargs = len(signature(func).parameters)
	x = np.linspace(start, end, bins)

	if nargs == 1:
		X = np.linspace(start, end, bins)
		Y = func(X)
		df_dx = dx(func)(X)
		df_dxdx = dx_dx(func)(X)

		fig = plt.figure(dpi=500)
		ax = fig.add_subplot(2, 2, 1, title='f(x)')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.plot(Y)
		b = fig.add_subplot(2, 2, 2, title='f\'(x)')
		b.set_xlabel('x')
		b.set_ylabel('y')
		b.plot(df_dx)
		c = fig.add_subplot(2, 2, 3, title='f\'\'(x)')
		c.set_xlabel('x')
		c.set_ylabel('y')
		c.plot(df_dxdx)
	elif nargs == 2:
		y = np.linspace(start, end, bins)

		X, Y = np.meshgrid(x, y)
		gradient = grad(func)
		gd = gradient(X, Y)
		Z = func(X, Y)

		fig = plt.figure(dpi=500)
		if plotfunc and plotgrad:
			ax = fig.add_subplot(1, 2, 1, projection='3d', title='f(x,y) = 2x^2 + xy + 2(y-3)^2', adjustable='box',
			                     aspect=1)
			ax.contour3D(X, Y, Z, 50, cmap='binary')
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
			b = fig.add_subplot(1, 2, 2, title='gradient', adjustable='box', aspect=1)
			b.quiver(X, Y, gd[0], gd[1])
		elif plotfunc:
			ax = fig.add_subplot(1, 1, 1, projection='3d', title='f(x,y) = 2x^2 + xy + 2(y-3)^2', adjustable='box',
			                     aspect=1)
			ax.contour3D(X, Y, Z, 50, cmap='binary')
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
		elif plotgrad:
			b = fig.add_subplot(1, 1, 1, title='gradient', adjustable='box', aspect=1)
			b.quiver(X, Y, gd[0], gd[1])
	plt.show()
	if savefig:
		if plotfunc and plotgrad:
			filename = 'plot'
		elif plotfunc:
			filename = 'function'
		elif plotgrad:
			filename = 'gradient'
		logging.info('saving plot as svg...')
		fig.savefig(filename + '.svg')


if __name__ == '__main__':
	# print(newton(lambda x, y: 2 * (x ** 2) + (x * y) + 2 * ((y - 3) ** 2), (-1, 4)))
	# plot(lambda x, y: 2 * (x ** 2) + (x * y) + 2 * ((y - 3) ** 2), savefig=True, plotfunc=False)
	# plot(lambda x, y: 2 * (x ** 2) + (x * y) + 2 * ((y - 3) ** 2), savefig=True, plotgrad=False)
	print(newton(lambda x: x ** 2, 2))
	plot(lambda x: x ** 2)
