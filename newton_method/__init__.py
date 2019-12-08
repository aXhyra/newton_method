from inspect import signature
import numpy as np
import logging
from mpl_toolkits import mplot3d


def df_dx(func, epsilon=1e-3):
	nargs = len(signature(func).parameters)
	# 3D function
	if nargs == 2:
		return lambda x, y: ((func(x + epsilon, y) - func(x, y)) / epsilon)
	# 2D function
	if nargs == 1:
		return lambda x: ((func(x + epsilon) - func(x)) / epsilon)
	logging.error('Cannot work on functions with more than 2 variables')


def df_dy(func, epsilon=1e-3):
	return lambda x, y: ((func(x, y + epsilon) - func(x, y)) / epsilon)


def df_dxdx(func):
	nargs = len(signature(func).parameters)
	# 3D function
	if nargs == 2:
		return lambda x, y: df_dx(df_dx(func))(x, y)
	# 2D function
	if nargs == 1:
		return lambda x: df_dx(df_dx(func))(x)
	logging.error('Cannot work on functions with more than 2 variables')


def df_dydy(func):
	return lambda x, y: df_dy(df_dy(func))(x, y)


def df_dxdy(func):
	return lambda x, y: df_dx(df_dy(func))(x, y)


def df_dydx(func):
	return lambda x, y: df_dy(df_dx(func))(x, y)


def grad(func):
	nargs = len(signature(func).parameters)
	if nargs == 2:
		return lambda x, y: np.array([df_dx(func)(x, y), df_dy(func)(x, y)])
	logging.error('Cannot calculate gradient, function is not 3D')


def hess(func):
	nargs = len(signature(func).parameters)
	if nargs == 2:
		return lambda x, y: np.array([[df_dxdx(func)(x, y), df_dxdy(func)(x, y)],
		                              [df_dydx(func)(x, y), df_dydy(func)(x, y)]])
	logging.error('Cannot calculate hessian, function is not 3D')


def newton(func, x0=(0, 0), steps=2, delta=.05):
	nargs = len(signature(func).parameters)

	# 3D function
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

	# 2D function
	elif nargs == 1:
		res = x0
		dx = df_dx(func)
		dxdx = df_dxdx(func)

		for i in range(steps):
			prev = res
			res = prev - dx(prev) / dxdx(prev)
			diff = abs(prev - res)
			if diff < delta:
				break

	return res


def plot(func, start=-10, end=10, bins=100, savefig=False, plotfunc=True, plotgrad=True, filename=''):
	from matplotlib import pyplot as plt
	nargs = len(signature(func).parameters)
	x = np.linspace(start, end, bins)

	# if given function is 2D
	if nargs == 1:
		X = np.linspace(start, end, bins)
		Y = func(X)
		dx = df_dx(func)(X)
		dxdx = df_dxdx(func)(X)

		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(2, 2, 1, title='f(x)')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.plot(Y)
		b = fig.add_subplot(2, 2, 2, title='f\'(x)')
		b.set_xlabel('x')
		b.set_ylabel('y')
		b.plot(dx)
		c = fig.add_subplot(2, 2, 3, title='f\'\'(x)')
		c.set_xlabel('x')
		c.set_ylabel('y')
		c.plot(dxdx)

	# 3D function
	elif nargs == 2:
		y = np.linspace(start, end, bins)

		X, Y = np.meshgrid(x, y)
		gradient = grad(func)
		gd = gradient(X, Y)
		Z = func(X, Y)

		fig = plt.figure(figsize=(8, 6))
		if plotfunc and plotgrad:
			ax = fig.add_subplot(1, 2, 1, projection='3d', title='f(x,y)', adjustable='box', aspect=1)
			ax.contour3D(X, Y, Z, 50, cmap='binary')
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
			b = fig.add_subplot(1, 2, 2, title='gradient', adjustable='box', aspect=1)
			b.quiver(X, Y, gd[0], gd[1])

		elif plotfunc:
			ax = fig.add_subplot(1, 1, 1, projection='3d', title='f(x,y)', adjustable='box', aspect=1)
			ax.contour3D(X, Y, Z, 50, cmap='binary')
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')

		elif plotgrad:
			b = fig.add_subplot(1, 1, 1, title='gradient', adjustable='box', aspect=1)
			b.quiver(X, Y, gd[0], gd[1])

	plt.show()

	if savefig and plotfunc or plotgrad:
		if plotfunc and plotgrad:
			filename = 'plot'
		elif plotfunc:
			filename = 'function'
		elif plotgrad:
			filename = 'gradient'

		logging.info('saving plot as svg...')
		fig.savefig(filename + '.svg')


if __name__ == '__main__':
	np.set_printoptions(precision=3)
	print(newton(lambda x, y: 2 * (x ** 2) + (x * y) + 2 * ((y - 3) ** 2), (-1, 4)))
	plot(lambda x, y: 2 * (x ** 2) + (x * y) + 2 * ((y - 3) ** 2), plotfunc=False)
	plot(lambda x, y: 2 * (x ** 2) + (x * y) + 2 * ((y - 3) ** 2), plotgrad=False)
	print('%.2f' % newton(lambda x: x ** 2, 2))
	plot(lambda x: x ** 2)
