import logging
import torch

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.endmodel import MLPModel
from wrench.labelmodel import MajorityVoting
import numpy as np
from datetime import datetime
import os
import argparse
import random


def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    argparser.add_argument('--seed', type=int, default=921, help='random seed')
    argparser.add_argument('--dataset_home', type=str, default='datasets', help='dataset path')
    argparser.add_argument('--dataset', type=str, default='yelp', help='dataset name')
    argparser.add_argument('--dataset_ar', type=str, default='datasets_1', help='dataset ar name')
    argparser.add_argument('--tf', type=str, default='train_ud', help='training file')
    argparser.add_argument('--ef', type=str, default='test', help='evaluating file')
    argparser.add_argument('--n_steps', type=int, default=20000, help='training steps')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--test_batch_size', type=int, default=1000, help='tests batch size')
    argparser.add_argument('--patience', type=int, default=50, help='patience')
    argparser.add_argument('--evaluation_step', type=int, default=20, help='evaluation step')
    argparser.add_argument('--metric', type=str, default='acc', help='evaluation metric')
    # argparser.add_argument('--extract_fn', type=str, default='bert', help='extract_fn')
    # argparser.add_argument('--model_name', type=str, default='bert', help='bert-base-cased')

    args = argparser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

    return


args = parser()
print(args)
setup_seed(args.seed)
start_time = datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Extract data features using pre-trained BERT model and cache it
extract_fn = 'bert'
model_name = 'bert-base-cased'

if args.dataset=='agnews':
    train_data, valid_data, test_data = load_dataset(args.dataset_home, args.dataset, args.dataset_ar, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name, tf=args.tf, ef=args.ef, device=device)
elif args.dataset=='yelp':
    train_data, valid_data, test_data = load_dataset(args.dataset_home, args.dataset, args.dataset_ar, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name, tf=args.tf, ef=args.ef, device=device)
elif args.dataset=='basketball':
    train_data, valid_data, test_data = load_dataset(args.dataset_home, args.dataset, args.dataset_ar, extract_feature=False, tf=args.tf, ef=args.ef, device=device)

elif args.dataset=='tennis':
    train_data, valid_data, test_data = load_dataset(args.dataset_home, args.dataset, args.dataset_ar, extract_feature=False, tf=args.tf, ef=args.ef, device=device)

#### Generate soft training label via a label model
#### The weak labels provided by supervision sources are alreadly encoded in dataset object
label_model = MajorityVoting(device=device)
label_model.fit(train_data, valid_data)
soft_label = label_model.predict_proba(train_data)

#### Train a MLP classifier with soft label
model = MLPModel(n_steps=args.n_steps, batch_size=args.batch_size, test_batch_size=args.test_batch_size)
history = model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=soft_label, 
                    device=device, metric=args.metric, patience=args.patience, evaluation_step=args.evaluation_step)

#### Evaluate the trained model
metric_value = model.test(test_data, args.metric)
print(metric_value)

end_time = datetime.now()
print("start time:", start_time)
print("end time:", end_time)
print("lasting time:", end_time-start_time)
