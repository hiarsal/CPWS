from cpws import CPWS

cpws = CPWS()

cpws.add_dict_key('datasets/yelp/datasets_1/train_ud_wo_ab0.json', 'datasets/yelp/datasets_1/train_ud_wo_ab0.npy',
             'datasets/yelp/datasets_1/train_ud_wo_ab0_cs.json')

cpws.model_lf_clean(fin='datasets/yelp/datasets_1/train_ud_wo_ab0_cs.json',
               fout='datasets/yelp/datasets_1/train_ud_wo_ab0_cs_clean.json')

print(count_class_rate(fp='datasets/yelp/datasets_1/train_ud_wo_ab0_cs_clean.json',
                       fd_name='yelp'))
