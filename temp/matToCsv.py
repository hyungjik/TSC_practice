import scipy.io
import pandas as pd
import os

print(os.listdir('.'))
mat = scipy.io.loadmat('Libras.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.iteritems()})
data.to_csv("example.csv")
