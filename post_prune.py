from cpws import CPWS
import numpy as np

cpws = CPWS()

P_ = np.array([[0.7242, 48.1231], [0.5202, 0.3457], [0.6494, 0.4315], [0.6123, 0.4069], [0.5942, 0.3948], [0.5543, 0.3683], [0.5877, 0.3905], [0.6289, 0.4179]])

r = 2.1
ratio = r/P_[0][0]
P = P_*ratio

for i in range(cpws.get_n_LFs('yelp')):
  cpws.prune_(prop=P[i], wd='datasets/yelp/datasets_1/train_ud_wo_ab%d_cs_clean.json'%i,
        fp='datasets/yelp/datasets_1/train_ud_wo_ab%d_cs_clean_pr.json'%i, fd_name='yelp')
