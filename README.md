newton_method
=============

## Introduction
This package contains functions to calculate the minimum of a mathematical function written as a lambda using Newton's
gradient method (https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization).

## Functions
This package contains the following functions:
* df_dx(func) returns a lambda function of the partial derivative on x direction
* df_dy(func) returns a lambda function of the partial derivative on y direction
* df_dxdx(func) returns a lambda function of the double partial derivative on x direction
* df_dxdy(func) returns a lambda function of the mixed double partial derivative on x and y directions
* df_dydy(func) returns a lambda function of the double partial derivative on y direction
* df_dydx(func) returns a lambda function of the mixed double partial derivative on y and x directions
* grad(func) returns an array of lambda functions (gradient of the function func)
* hess(func) returns a matrix of lambda functions (hessian of the function func)
* newton(func, x0=(0, 0), steps=2, delta=.05) returns the minimum point of the function from the point x0 with steps
steps or until the delta is under delta
* plot(func, start=-10, end=10, bins=100, savefig=False, plotfunc=True, plotgrad=True, filename='') plots the function
func with its gradient if its a 2 variable function, in case it's a 2D function plots the function, its first and second
derivative

func has to be a lambda function, for example: 

    df_dx(lambda x,y: x ** 2 + y ** 2)

## Installation
#### pip:
install this package using pip:

    pip install newton_method
#### Manual install:
clone this repo and run setup.py
 
    git clone https://github.com/aXhyra/newton_method
    cd newton_method
    pip install .

