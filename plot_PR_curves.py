#sed 's/.*P_dev: \(.*\), R_dev: \(.*\), F.*/\1 \2/' results_tmp.txt > results_tmp2.txt

import matplotlib.pyplot as plt
import numpy as np
import pdb

f = open('results_tmp2.txt')
A = []
for line in f:
  line = line.rstrip('\n')
  fields = line.split()
  a = [float(val) for val in fields]
  A.append(a)

f.close()

A_soft = A[:10]
A_sparse = A[10:]

X_soft = np.array(A_soft)
X_sparse = np.array(A_sparse)

plt.plot(X_soft[:,0], X_soft[:,1], 'b.', X_sparse[:, 0], X_sparse[:,1], 'ro')
plt.show()

pdb.set_trace()
