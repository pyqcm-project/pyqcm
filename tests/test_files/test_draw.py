import pyqcm
from model_graphene_bath import model

import matplotlib.pyplot as plt


# model.draw_operator('t', show_neighbors=True, values=True, show_labels=True, plt_ax=plt.gca())
# plt.savefig('draw_t.pdf')
# plt.cla()

# model.draw_operator('U', show_neighbors=True, values=True, show_labels=True, plt_ax=plt.gca())
# plt.savefig('draw_U.pdf')
# plt.cla()

model.draw_operator('M', show_neighbors=True, values=True, show_labels=True, plt_ax=plt.gca())
plt.savefig('draw_M.pdf')
plt.cla()

model.draw_operator('Mx', show_neighbors=True, values=True, show_labels=True, plt_ax=plt.gca())
plt.savefig('draw_Mx.pdf')
plt.cla()

model.draw_operator('cdw', show_neighbors=True, values=True, show_labels=True, plt_ax=plt.gca())
plt.savefig('draw_cdw.pdf')
plt.cla()

model.draw_cluster_operator(0, 'tb1', values=True, show_labels=True, plt_ax=plt.gca())
plt.savefig('draw_tb1.pdf')


