import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
  
  f_integrals_small = "small/integrals_small_concurrent.txt"
  data = np.genfromtxt(f_integrals_small, delimiter=",", comments='#',skip_header=5)
  
  fig,ax = plt.subplots()
  colors = ['r','b','g']
  threads = [4,6,8]
  
  labels = [f"{x} Threads" for x in threads]
  
  for j in range(1,2):#data.shape[1]):
      cond = ~ np.isnan(data[:,j])
      Y = data[:,j][cond]
      X = data[:,0][cond]
      ax.plot(X, Y, color=colors[j-1], label=labels[j-1], marker='o')
  
  ax.set_xlabel("Concurrent instance (proportional to server load)")
  ax.set_ylabel("Median time (s)")
  ax.grid()
  ax.set_ylim(10,45)
  ax.set_xlim([1,24])
  #ax.set_yscale("log", base=2)
  #ax.set_xscale("log", base=2)
  #ax.set_yticks([int(2**x) for x in range(2,7)])
  #ax.set_yticklabels([str(int(x)) for x in ax.get_yticks()])
  #ax.set_xticks([int(2**x) for x in range(0,5)])
  #ax.set_xticklabels([str(int(x)) for x in ax.get_xticks()])
  ax.legend()
  
  plt.tight_layout()
  plt.show()

