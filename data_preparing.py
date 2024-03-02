from cpws import CPWS


cpws = CPWS()

# =============data_split===============
ds_v = cpws.load_json("datasets/yelp/train.json")
len_train = len(ds_v)
print(len_train)

AR = 0.01
rd, ud = cpws.data_split(ds_v, AR)
cpws.save_json(rd, 'datasets/yelp/datasets_1/train_rd.json')
cpws.save_json(ud, 'datasets/yelp/datasets_1/train_ud.json')

#============weak_data_gen=======
ds_v = cpws.load_json('datasets/yelp/datasets_1/train_ud.json')
weak_datasets = cpws.weak_data_gen(ds_v, fd_name='yelp')
for i in range(len(weak_datasets)):
    # print(len(weak_datasets[i]))
    cpws.save_json(weak_datasets[i], 'datasets/yelp/datasets_1/train_ud_wo_ab%d.json'%i)

# ============weak_data_gen_full=======
ds_v_ = cpws.load_json('datasets/yelp/datasets_1/train_ud.json')
weak_datasets = cpws.weak_data_gen_full(ds_v_, fd_name='yelp')
for i in range(len(weak_datasets)):
    # print(len(weak_datasets[i]))
    cpws.save_json(weak_datasets[i], 'datasets/yelp/datasets_1/train_ud_w_ab%d.json'%i)
