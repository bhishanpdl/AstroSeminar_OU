import scipy.optimize as spo

def func(x,a,b,c):
	return a*x**2 + b*x  + c

minimizer_kwargs = {'args': (3,4,5)}
res = spo.basinhopping(func,x0=[1.],minimizer_kwargs=minimizer_kwargs,niter=100)

print(res)
print("global minimum: x = %.4f,    f(x0) = %.4f" % (res.x, res.fun))
print("  WolframAlpha: x = -0.66667, f(x0) = 3.6667 ")