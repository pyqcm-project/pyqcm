import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  
  f_integrals_small = "bench_ED2_laptop.txt"
  data = np.genfromtxt(f_integrals_small, delimiter=",", comments='#',skip_header=6)
  
  fig,ax = plt.subplots()
  colors = ['r','b','g']
  
  for j in range(1,2):#data.shape[1]):
      ax.plot(data[:,0], data[:,j], color=colors[j-1], label="pyqcm", marker='o')
  
  max_core = int(data[-1,0])
  ref_time = data[0,1]
  ax.plot([x+1 for x in range(max_core)],[ref_time/(x+1) for x in range(max_core)], linestyle='--', color='k', label='Ideal')
  
  ax.set_xlabel("Cores")
  ax.set_ylabel("Time (s)")
  ax.grid()
  ax.set_ylim(64,660)
  ax.set_xlim([1,8])
  ax.set_yscale("log", base=2)
  ax.set_xscale("log", base=2)
  ax.set_yticks([int(2**x) for x in range(6,10)])
  ax.set_yticklabels([str(int(x)) for x in ax.get_yticks()])
  ax.set_xticks([int(2**x) for x in range(0,4)])
  ax.set_xticklabels([str(int(x)) for x in ax.get_xticks()])
  ax.legend()
  
  plt.tight_layout()
  plt.show()

