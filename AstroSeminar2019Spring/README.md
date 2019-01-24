# Astro seminar 2019 Spring (Jan to May)

# Papers to discuss
- [Data analysis recipes: Fitting a model to data](https://arxiv.org/abs/1008.4686)
- [Dos and don’ts of reduced chi-squared](https://arxiv.org/abs/1012.3754)
- [Error estimation in astronomy: A guide](https://arxiv.org/abs/1009.2755)


# Paper Data Analysis Recipes
![](images/linear_regression_dar.png)
```python
# Load a dataset with first 5 rows as outliers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [10, 8]
%matplotlib inline

import scipy.linalg as linalg

# load data
df = pd.read_csv('data_allerr.dat',sep='&')
df.columns = [i.strip('#').strip() for i in df.columns]
print(df.shape)
df.head(10)

# matrices A,C, Y, yerr
df1 = df.iloc[4:, :]
x = df1.x.values
y = df1.y.values
yerr = df1['sigm_y'].values  # sigma y is yerr

degree = 2
Y = y
A = np.vander(x, degree+1, increasing=True).astype(float)  # 1, x, x**2
C = np.diag(yerr*yerr)  # diagonal matrix of yerr**2

# Bestfit
cinv = linalg.inv(C)
cinv_y = cinv @ Y.T
at_cinv_y = A.T @ cinv_y

cinv_a = cinv @ A
at_cinv_a = A.T @ cinv_a

bestfitvar = linalg.inv(at_cinv_a)
bestfit = bestfitvar @ at_cinv_y  # bestfit = params = c,b,a for ax**2 + bx + c

# plot bestfit
xrange = [0, 300]
yrange = [0, 700]
nsamples = 1001
xs = np.linspace(xrange[0], xrange[1], nsamples)
ys = np.zeros(len(xs))
for i in range(len(bestfit)):
    ys += bestfit[i] * xs**i

# plot
plt.plot(xs, ys, 'k-')
plt.xlim(xrange)
plt.ylim(yrange)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.errorbar(x, y, yerr, marker='o',color='k', linestyle='None')

# print text
# reverse the bestfit and bestfitvar
params = bestfit.tolist()[::-1]
err = np.sqrt(np.diag(bestfitvar)).tolist()[::-1]

# alternative params and err
params_err = [None]*(len(params)+len(err))
params_err[::2] = params
params_err[1::2] = err

# format text
fmt = [ ('({:.2g} \pm {:.2g})x^' + str(i) +' + ') for i in reversed(range(len(params)))]
fmt = ''.join(fmt)
fmt = fmt.rstrip('x^0 + ').replace('x^1','x')
text = r'$y = ' + fmt.format(*params_err) + r'$'

# plot text
plt.text(5, 30, text)
plt.show()
```
![](images/DAR_ex3.png)
