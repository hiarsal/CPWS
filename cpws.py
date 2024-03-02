import torch
import numpy as np
import random
import json
import copy
import torch.nn.functional as F


class CPWS:
    def __init__(self) -> None:
        super(CPWS, self).__init__()

    def pos_label(self, ds):
        """
        Function:
            Get positive label(e.g., 0 or 1) for different LFs and datasets
        Arguments:
            ds: dataset name
        Return:
            v_pos: vector of positive label for the given dataset. 
            -1 means non-certain specific positive label
        """
        if ds == 'basketball':
            # lf_3: w/o abstain in fact
            v_pos = [0, 1, 1, -1]
        elif ds == 'yelp':
            # only lf_1 has one label, others has two
            v_pos = [-1, 1, -1, -1, -1, -1, -1, -1]
        elif ds == 'agnews':
            v_pos = [0, 0, 1, 1, 1, 3, 3, 2, 2]
        elif ds == 'tennis':
            v_pos = [-1, -1, -1, -1, -1, -1]
        return v_pos

    def get_n_LFs(self, ds):
        """
        Function:
            get the number of LFs for dataset ds
        Arguments:
            ds: dataset name
        Return:
            n_LFs: number of LFs for dataset ds
        """
        if ds == 'yelp':
            n_LFs = 8
        elif ds == 'agnews':
            n_LFs = 9
        elif ds == 'tennis':
            n_LFs = 6
        elif ds == 'basketball':
            n_LFs = 4
        
        return n_LFs


    def get_n_cls(self, ds):
        """
        Function:
            get the number of classes for dataset ds
        Arguments:
            ds: dataset name
        Return:
            n_cls: number of LFs for dataset ds
        """
        n_cls = 0
        if ds == 'yelp':
            n_cls = 2
        elif ds == 'agnews':
            n_cls = 4
        elif ds == 'tennis':
            n_cls = 2
        elif ds == 'basketball':
            n_cls = 2
        
        return n_cls


    def data_split(self, ds, ar):
        """
        Function:
            Split dataset ds into two parts with the proportion of prop, 
            i.e. labeled(x^{gt}) and unlabeled data(x^u)
        Arguments:
            ds: dataset, ordereddict
            ar: anotated data proportion, %-like for the first(x^{gt}), the remaining(x^u) is (1-prop)
            seed: random seed
        Return:
            labeled_data: reliable data(x^{gt}) with proportion of ar
            unlabeled_data: unreliable data(x^u) with proportion (1-ar)
        """
        len_ds = len(ds)
        # len_gt: scale for reliable learning, len_weak: scale for unreliable learning
        len_gt = int(len_ds*ar)
        # len_weak = len_ds-len_gt

        # print(len_gt)
        ds_list = [i for i in range(len_ds)]
        # ds_list = np.array(range(len_ds))
        sample_list = random.sample(ds_list, len_gt)
        # print(sample_list)

        labeled_data = {key: value for key, value in ds.items() if int(key) in sample_list}

        unlabeled_data = {key: value for key, value in ds.items() if int(key) not in sample_list}

        # 
        return labeled_data, unlabeled_data

    def save_json(self, fd, fp):
        """
        Function:
            Save data into a json file
        Arguments:
            fd: dict file
            fp: file path
        """
        js = json.dumps(fd)
        with open(fp, 'w') as jf:
            jf.write(js)
        jf.close()


    def load_json(self, fp):
        """
        Function:
            Load data from a json file
        Arguments:
            fp: file path
        Return:
            fd: loaded data
            jf: file pointer, to be closed out of the func
        """
        with open(fp) as jf:
            fd = json.load(jf)
        
        return fd


    def weak_data_gen(self, fd, fd_name):
        """
        Function:
            generate weak datasets by different LFs from dataset fd, delete abstain
        Arguments:
            fd: sorted dict, data
            fd_name: dataset name, e.g., youtube, basketball, imdb
        Return:
            weak_datasets: n_LFs weak datasets
        """
        n_LFs = self.get_n_LFs(fd_name)
        # n_LFs copoies of datasets
        weak_datasets = []
        for i in range(n_LFs):
            fd_ = copy.deepcopy(fd)
            weak_datasets.append(fd_)

        for i in range(n_LFs):
            for id in list(weak_datasets[i].keys()):
                if weak_datasets[i][id]['weak_labels'][i] == -1:
                    del weak_datasets[i][id]
                else:
                    weak_datasets[i][id]['label'] = weak_datasets[i][id]['weak_labels'][i]
            print("len of", i, "-th weak datset is:", len(weak_datasets[i]))
        return weak_datasets    


    def weak_data_gen_full(self, fd, fd_name):
        """
        Function:
            generate sevaral weak datasets by different LFs from dataset fd for unreliable learning
            for each LF, treat abstain as negtive
        Arguments:
            fd: data
            fd_name: dataset name, e.g., youtube, basketball, imdb
        Return:
            weak_datasets: n_LFs weak datasets
        """
        n_LFs = self.get_n_LFs(fd_name)
        v_pos = self.pos_label(fd_name)
        n_classes = self.get_n_cls(fd_name)

        # n_LFs copoies of datasets
        weak_datasets = []
        for i in range(n_LFs):
            fd_ = copy.deepcopy(fd)
            weak_datasets.append(fd_)

        for i in range(n_LFs):
            for id in list(weak_datasets[i].keys()):
                if weak_datasets[i][id]['weak_labels'][i] == -1:
                    # set abstain as negtive sample contrastive to the positive one
                    if v_pos[i]!=-1:
                        weak_datasets[i][id]['label'] = 1-v_pos[i]
                    else:
                        weak_datasets[i][id]['label'] = int(random.choice(np.arange(n_classes)))
                    # del weak_datasets[i][id]
                else:
                    weak_datasets[i][id]['label'] = weak_datasets[i][id]['weak_labels'][i]

            print("len of", i, "-th weak datset is:", len(weak_datasets[i]))
        return weak_datasets


    def prune_proportion(self, prop, wd, fd_name, fp):
        """
        Function:
            rank and prune data with a %-form proportion
        Arguments:
            prop: a vector or a scalar, denoting the number to prune(reserve) for each class
                    decided by data numbers of each class and LF
            wd: weak data with confidence score 'cs', json file, path(str)
            fd_name: dataset name, e.g., youtube, basketball, imdb
            fp: file path to save generated clean data
        Return:
            None, saving 'confident' data: cd
        """
        wd_dict = self.load_json(wd)
        for key in wd_dict.keys():
            wd_dict[key]['cs'] = max(wd_dict[key]['cs'])
        # new n_cls dicts
        cds = []
        n_cls = self.get_n_cls(fd_name)
        for i in range(n_cls):
            cds.append(dict())
        # form n_cls data dicts
        for key in wd_dict.keys():
            cds[wd_dict[key]['label']][key] = wd_dict[key]
        for i in range(n_cls):
            # sort data according to confidence score with descending order 
            cds[i] = sorted(cds[i].items(), key=lambda x:x['cs'], reverse=True)
            for j in range(prop[i]):
                cds[i].pop()
        # aggregate
        cd = {}
        for i in range(n_cls):
            cd.update(cds[n_cls])
        
        self.save_json(cd, fp)


    def prune_prop_cls(self, prop, cs, wd, fp, cls):
        """
        Function:
            rank and prune data with a %-form proportion for a single class/LF
        Arguments:
            prop: a value, given the proportion, select 'confident' samples according to their confidence score
            cs: confidence score, json file, path(str), {"id":cs}
            wd: weak data, json file, path(str)
            fp: file path to save generated clean data
            cls: label to be pruned
        Return:
            None, saving 'confident' data: cd
        """
        cs_dict = self.load_json(cs)
        wd_dict = self.load_json(wd)
        # prune numbers
        pn = (1-prop)*len(cs_dict)
        # get i-th class score
        for key in cs_dict:
            cs_dict[key] = cs_dict[key][cls]
        # sort data according to confidence score with descending order 
        cs_dict_sorted = sorted(cs_dict.items(), key=lambda x:x[1], reverse=True)
        # del last pn items 
        for i in range(pn):
            cs_dict_sorted.pop()
        # clean data and add confidence score
        for key in list(wd_dict.keys()):
            if key in cs_dict_sorted.keys():
                wd_dict[key]['cs'] = cs_dict_sorted[key]
            else:
                del wd_dict[key]
            
        self.save_json(wd_dict, fp)


    def prune(self, prop, wd, fp):
        """
        Function:
            rank and prune data with a %-form proportion for a single class/LF
        Arguments:
            prop: a value, given the proportion, select 'confident' samples according to their confidence score
            wd: weak data, json file, path(str)
            fp: file path to save generated clean data
        Return:
            None, saving 'confident' data: cd
        """    
        wd_dict = self.load_json(wd)
        for key in wd_dict.keys():
            wd_dict[key]['cs'] = wd_dict[key]['cs'][wd_dict[key]['label']]
        # sort data according to confidence score with descending order 

        wd_dict_sorted = sorted(wd_dict.items(), key=lambda x:x[1]['cs'], reverse=True)

        pn = (1-prop)*len(wd_dict_sorted)
        for i in range(int(pn)):
            wd_dict_sorted.pop()
        wd_dict_sorted_ = {}
        for item in wd_dict_sorted:
            wd_dict_sorted_[item[0]] = item[1]
        self.save_json(wd_dict_sorted_,fp)    


    def prune_(self, prop, wd, fp, fd_name):
        """
        Function:
            rank and prune data with a %-form proportion
        Arguments:
            prop: a vector, given the proportion,
                select 'confident' samples according to their confidence score for each class
            wd: weak data, json file, path(str)
            fp: file path to save generated clean data
            fd_name: dataset name, e.g., youtube, basketball, imdb
        Return:
            None, saving 'confident' data: cd
        """
        wd_dict = self.load_json(wd)
        for key in wd_dict.keys():
            wd_dict[key]['cs'] = wd_dict[key]['cs'][wd_dict[key]['label']]

        # new n_cls dicts
        cds = []
        n_cls = self.get_n_cls(fd_name)
        for i in range(n_cls):
            cds.append(dict())
        # form n_cls data dicts
        for key in wd_dict.keys():
            cds[int(wd_dict[key]['label'])][key] = wd_dict[key]

        for i in range(n_cls):
            # sort data according to confidence score with descending order 
            cds[i] = sorted(cds[i].items(), key=lambda x:x[1]['cs'], reverse=True)
            pn = (1-prop[i])*len(cds[i])
            for j in range(int(pn)):
                if len(cds[i])!=0:
                    cds[i].pop()
                else:
                    continue
        # aggregate
        cd = {}
        for i in range(n_cls):
            cd.update(cds[i])
        self.save_json(cd, fp) 


    def aggregation(self, fp_in, fp_out):
        """
        Function:
            integrate many dict files and save the new dict file
            aggregate dunplicated, conflict and different data-label
        Arguments:
            fp_in: list, str, file paths for many dicts
            fp_out: list, str, new generated file path for dict
        Return:
            None, save the new file
        """   
        d = {}
        for fp in fp_in:
            dfp = self.load_json(fp)
            for key in list(dfp.keys()):
                if key in list(d.keys()):
                    # rank and compare
                    if dfp[key]['cs']>d[key]['cs']:
                        d[key] = dfp[key]
                else:
                    d[key] = dfp[key]
        self.save_json(d, fp_out)


    def agg_count(self, fp_in, fd_name='agnews'):
        """
        Function:
            aggregate and count the class distribution of the aggregated class distribution
        Arguments:
            fp_in: list, str, file paths for many dicts, clean files
            fd_name: dataset name, e.g., youtube, basketball, imdb
        Return:
            cr: np vector, counting rate for each class
            c: np vector, counting for each class
            c_sum: sum
        """
        d = {}
        for fp in fp_in:
            dfp = self.load_json(fp)
            for key in list(dfp.keys()):
                if key in list(d.keys()):
                    # rank and compare
                    if dfp[key]['cs']>d[key]['cs']:
                        d[key] = dfp[key]
                else:
                    d[key] = dfp[key]
                    
        n_cls = self.get_n_cls(fd_name)
        c =np.zeros(n_cls)  # class: [0, 1, 2]

        for key in d.keys():
            label = d[key]['label']
            c[int(label)] = c[int(label)]+1
        c_sum = c.sum()
        cr = c/c_sum

        return cr, c, c_sum


    def count_class_rate(self, fp, fd_name):
        """
        Function:
            count the rate of each class in a dataset
        Arguments:
            fp: str, file path of the dataset
            fd_name: dataset name, e.g., youtube, basketball, imdb
        Return:
            cr: np vector, counting rate for each class
            c: np vector, counting for each class
        """
        n_cls = self.get_n_cls(fd_name)

        d = self.load_json(fp)
        c =np.zeros(n_cls)  # class: [0, 1, 2]

        for key in d.keys():
            label = d[key]['label']
            c[int(label)] = c[int(label)]+1
        c_sum = c.sum()
        cr = c/c_sum

        return cr, c, c_sum


    def loss_fn_file(self, fp_probs, fp_gt, metric='ce'):
        """
        Function:
            compute the loss of probs and labels from files, to evaluate the certainty of the LF
        Arguments:
            fp_probs: str, file path, storing predicted probs as npy file
            fp_gt: str, file path, storing ground truth as json file
            metric: str, loss fn, such as 'ce', 'mse'
        Return:
            loss: float, mean loss over all samples and classes
        """
        probs = np.load(fp_probs)
        js_gt = self.load_json(fp_gt)
        gt = []
        for ind in js_gt.keys():
            gt.append(int(js_gt[ind]['label']))
        
        ts_probs = torch.from_numpy(probs)
        ts_gt = torch.Tensor(gt).long()
        if metric=='ce':
            loss = F.cross_entropy(ts_probs, ts_gt)
        elif metric=='mse':
            ts_gt = F.one_hot(ts_gt, num_classes=2)
            loss = F.mse_loss(ts_probs, ts_gt)
        return loss


    def add_dict_key(self, fd, fn, fdn):
        """
        Function:
            add key-value items(fn) to a dict file(fd)
        Arguments:
            fd: str, dict file path
            fn: str, numpy file path
            fdn: str, new dict file path
        Return:
            None, save the new dict file
        """
        # load file
        js_fd = self.load_json(self, fd)
        np_fn = np.load(fn).astype(np.float64)
        # traverse the dict
        i = 0
        print('lens:', len(js_fd), np_fn.shape)
        for id in js_fd.keys():
            js_fd[id]['cs'] = list(np_fn[i])
            i = i + 1
        self.save_json(js_fd, fdn)

    
    def model_lf_clean(self, fin, fout):
        """
        Function:
            remove those data whose model prediction and lf output is inconsistent
        Args:
            fin: str, json file path, raw data
            fout: str, json file path, clean data
        Return:
            None, save the new json dataset file
        """
        dfin = self.load_json(fin)

        for key in list(dfin.keys()):
            if dfin[key]['label'] != dfin[key]['cs'].index(max(dfin[key]['cs'])):
                del dfin[key]
        
        self.save_json(dfin, fout)
        