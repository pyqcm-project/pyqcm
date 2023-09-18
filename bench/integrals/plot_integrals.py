import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
  
  f_integrals_small = "small/integrals_small_results.txt"
  data = np.genfromtxt(f_integrals_small, delimiter=",", comments='#',skip_header=4)
  
  fig,ax = plt.subplots()
  colors = ['r','b','g']
  
  labels = ["MKL","OPENBLAS","BLIS"]
  
  for j in range(1,data.shape[1]):
      ax.plot(data[:,0], data[:,j], color=colors[j-1], label=labels[j-1], marker='o')
  print(data)
  max_core = int(data[0,0])
  ref_time = data[-1,1]
  ax.plot([x+1 for x in range(max_core)],[ref_time/(x+1) for x in range(max_core)], linestyle='--', color='k', label='Ideal')
  
  ax.set_xlabel("Cores")
  ax.set_ylabel("Time (s)")
  ax.grid()
  ax.set_ylim(4,300)
  ax.set_xlim([1,96])
  ax.set_yscale("log", base=2)
  ax.set_xscale("log", base=2)
  ax.set_yticks([int(2**x) for x in range(2,9)])
  ax.set_yticklabels([str(int(x)) for x in ax.get_yticks()])
  ax.set_xticks([int(2**x) for x in range(0,8)])
  ax.set_xticklabels([str(int(x)) for x in ax.get_xticks()])
  ax.legend()
  
  plt.tight_layout()
  plt.show()

