import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.function as F
import h5py

from scipy.sparse import csr_matrix
from functools import partial

# --
# Prepprocessers

# TODO what does it do?
class IdentityPrep(nn.Module):
    def __init_(self, input_dim, n_nodes=None):
        """Example of preprocessor --doesn't do anything??"""
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

    def forward(self, ids, feats, leayers_idx =0):
        return feats


class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, embedding_dim=64):
        """adds node embeding"""
        super(NodeEmbeddingPrep, self).__init_-()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embeeding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes +1, embedding_dim=embedding_dim)

        # fully connected sigle forward layer
        self.fc = nn.Linear(embedding_dim, embedding_dim) # Affine transform for changing scale and location

    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.emedding_dim
        else:
            return self.embedding_dim

    # TODO What does thid NodeeEmddingPrep.forward() do??
    def forward(self, ids, feats, layer_idx=0):
        if layer_idx >0:
            ebs = self.embedding(ids)
        else:
            # Don't look at nodes's own embedding for prediction, or you will probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))
        embs = self.fc(embs)
        if self.input_dim:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs

class LinearPrep(nn.Module):
    def __init__(self,input_dim, n_nodes, output_dim=32):
        """ add node embedding"""
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, idx, feats, layer_idx = 0):
        return self.fc(feats)


prep_lookup = {
    "idenity" : IdentityPrep,
    "node_embeding" : NodeEmbeddingPrep,
    "linear" : LinearPrep,
}


class AggregatorMixin(object):
    #property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1) # (1, output*2)

class MeanAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, combine_fn = lambda x:torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.output_dim = output_dim
        self.ativation = activation
        self.combine_fn = combine_fn

    # what is the dim of neibs? is it (# node, # feature, # neigh )
    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))# !!Careful
        agg_neib = agg_neib.mean(dim=1) #Careful

        out = self.combine_fn([self.fc_x(x), self.fc_neigh(agg_neib)])
        if self.ativation:
            out = self.actiation(out)
        return out


class PoolAggregator(nn.Module, AggregatorMixin):

    def __init__(self, input_dim, output_dim, pool_fn, activation, hidden_dim=512, combine_fn=lambda x:torch.cat(x, dim=1)):
        super(PoolAggregator, self).__init__()

        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias= True)
            nn.ReLu()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)

        self.output_dim = output_dim
        self.activation =activation
        self.pool_fn = pool_fn
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = self.pool_fn(agg_neib)

        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.actiation:
            out = self.activation(out)
        return out

class MaxPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation ,hidden_dim=512, combine_fn=lambda x:torch.cat(x, dim=1)):
        super(MaxPoolAggregator, self).__init__(**{
            "input_dim":input_dim,
            "output_dim":output_dim,
            "pool_fn": lambda x: x.max(dim=1)[0],
            "activation" : activation,
            "hidden_dim" :hidden_dim,
            "combine_fn":combine_fn
        })

class MeanPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation ,hidden_dim=512, combine_fn=lambda x:torch.cat(x, dim=1)):
        super(MeanPoolAggregator, self).__init__(**{
            "input_dim":input_dim,
            "output_dim":output_dim,
            "pool_fn": lambda x: x.mean(dim=1)[0],
            "activation" : activation,
            "hidden_dim" :hidden_dim,
            "combine_fn":combine_fn
        })

class LSTMAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, bidirectional=False, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(LSTMAggregator, self).__init__()
        not hidden_dim %2, "LSTMAggregator: hidden_dim %2 !=0 "

        self.lstm = nn.LSTM(input_dim, hidden_dim//(1+bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)

        self.output_dim = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        x_emb = self.fc_x(x)

        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:,-1,:] # !! Taking final state, but oculd do something better (eg. attaneion)
        neib_emb = self.fc_neib(agg_neib)

        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)
        return out


class AttnentionAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, hidden_dim = 32, combine_fn = lambda x: torch.cat(x, dim=1)):
        super(AttnentionAggregator, self).__init__()

        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            # why is ther nother linear layer added after Tahn() ??
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        ])
        self.fc_x = nn.Linear(input_dim, output_dim,bias= False)
        self.gc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):

        #not sure how attention works in aggregator layers try to come up with my own
        neib_att = self.att(neibs)
        x_att = self.att(x)
        neib_att = neib_att.view(x.size(0),-1, neib_att.size(1))
        x_att = x_att.view(x.size(0), x_att.size(1), 1)
        ws = F.softmax(torch.bmm(torch.bnn(neib_att, x_att).squeeze()))

        # Weihted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)

        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        return out


aggregator_lookup = {
    "mean" : MeanAggregator,
    "max_pool": MaxPoolAggregator,
    "mean_pool": MeanPoolAggregator,
    "lstm": LSTMAggregator,
    "attention": AttnentionAggregator
}


class UniformNeighborSampler(object):
    """
        Sample from a "dense 2D edgelish", which looks like
            [
                [1,2,3,...,1],
                [1,2,3,...,3],
                ...
            ]
    """

    def __init__(self,adj):
        self.adj = adj

    def __call__(self, ids, n_samples =-1):
        tmp = self.adj[ids]
        perm = torch.randperm
        if ids.is_cuda:
            perm = perm.cuda()

        tmp = tmp[:,perm] # shuffle
        return tmp[:,:n_samples]


sampler_lookup = {
    "uniform_neighbor_smapler": UniformNeighbourSampler,
    "sparse_uniform_neighbour_sampler" : SparseUniformNeihbourSampler,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--no-cuda', action="store_true")

    # Optimization params
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr-init', type=float, default=0.01)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=0.0)

    # Architecture params
    parser.add_argument('--sampler-class', type=str, default='uniform_neighbor_sampler')
    parser.add_argument('--aggregator-class', type=str, default='mean')
    parser.add_argument('--prep-class', type=str, default='identity')

    parser.add_argument('--n-train-samples', type=str, default='25,10')
    parser.add_argument('--n-val-samples', type=str, default='25,10')
    parser.add_argument('--output-dims', type=str, default='128,128')

    # Logging
    parser.add_argument('--log-interval', default=10, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--show-test', action="store_true")

    # --
    # Validate args

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available
    assert args.prep_class in  prep_lookup.keys(), 'parse_args: prep_class not in %s' % str(prep_lookup.keys())
    assert args.aggregator_class in aggregator_lookup.keys(), 'parse_args: agregator_class not in %s' % str(aggregator_lookup.keys())
    assert args.batch_size >1, 'parse_args: batchsize must be >1'
    return args


def parse_csr_matrix(x):
    v, r, c = x
    return csr_matrix((v, (r,c)))

class NodeProblem(object):
    def __init__(self, problem_path, cuda = True ):
        print('NodepROblems: loading started')

        f = h5py.File(problem_path)
        self.task = f['task']
        self.task      = f['task'].value
        self.n_classes = f['n_classes'].value if 'n_classes' in f else 1 # !!
        self.feats     = f['feats'].value if 'feats' in f else None
        self.folds     = f['folds'].value
        self.targets   = f['targets'].value

        if 'sparse' in f and f['sparse'].value:
            self.adj = parse_csr_matrix(f['adj'].value)
            self.train_adj = parse_csr_matrix(f['train_adj'].value)
        else:
            self.adj = f['adj'].value
            self.train_adj = f['train_adj'].value

        f.close()

        self.fetas_dim = self.feats.shape[1] if self.feats is not None else None
        self.n_nodes = self.adj.shape[0]
        self.cuda = cuda
        self.__to_torch()

        self.nodes = {
            "train" : np.where(self.folds == 'train')[0],
            "val" : np.where(self.folds == "val")[0],
            "test" : np.where(self.folds == "test")[0]
        }

        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)

        print('NodeProblm" loading finished')

    def __to_torch(self):
        if not sparse.issparse(self.adj):
            self.adj = Variable(torch.LongTensor(self.adj))
            self.train_adj = Variable(torch.LongTensor(self.train_adj))
            if self.cuda:
                self.adj = self.adj.cuda()
                self.train_adj = self.train_adj.cuda()

        if self.feats is not None:
            self.feats = Variable(torch.FloatTensor(self.feats))
            if self.cuda:
                self.feats = self.feats.cuda()

    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]

        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)

        n_chunks = idx.shape[0] // batch_size +1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[n_chunks]
            targets = self.targets[mids] # target = label ?
            mids, tagets = self.__batch_to_torch(mids, targets)
            yield mids, targets, chunk_id / n_chunks

    def __batch_to_torch(self, mids, targets):
        """conert batch to torch"""
        mids = Variable(torch.LongTensor(mids))

        if self.task == "multilabel_calssifciation":
            targets = Variable(torch.FloatTensor(targets))
        elif self.task == "classificaiotn:":
            targets = Variable(torch.LongTensor(targets))
        elif 'regression' in self.task:
            targets = Variable(troch.FloatTensor(targets))
        else:
            raise Exception("NodeDataLoader: unkown task: %s" % self.task)


class GSSupervised(nn.Module):
    def __init__(self,
        input_dim,
        n_nodes,
        n_classes,
        layer_specs,
        aggregator_class,
        prep_class,
        sampler_class, adj, train_adj,
        lr_init = 0.01,
         weight_decay = 0.0,
         lr_schedule="constant",
         epochs=10):

        super(GSSupervised, self).__init__()

        #Define network

        #Sampler
        self.train_sampler = sampler_class(adj=train_adj)
        self.val_sapler = sampler_class(adj=adj)
        self.train_sample_fns = [partial(self.train_sapmler, n_samplers= s['n_train_sampler']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_sampler=s['n_val_samples']) for s in layer_specs]

        # Prep
        self.prep = prep_class(input_dim=input_dim, n_nodes=n_nodes)
        input_dim = self.prep.output_dim

        #Network
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(
                input_dim=input_dim,
                output_dim=spec['output_dim'],
                actiation=spec['activtion'],
            )
            agg_layers.append(agg)
            input_dim = agg.output_dim # Mynot be the same as spec['output_dim']
            self.agg_layers = nn.Sequenctial(*agg_layers)

        self.fc = nn.Linear(input_dim, n_classes, bias=True)
        # --
        # Define optimizer

        self.lr_scheduler = partial(getattr(LRSchedule, lr_schedule), lr_init=lr_init)
        self.lr = self.lr_scheduler(0.0)
        self.optimzier = torch.optim.Adam(self.parameters(), lr=self.lr, weight_deecay=weight_decay)

    def forward(self, ids, feats, train):
        # Sample neighbors
        sample_fns = self.train_sample_fns if train else self.val_sample_fns

        has_feats = feats is not None
        tmp_feats = feats[ids] if has_feats else None
        all_feats = [self.prep(ids, tmp_feats, layer_idx=0)]
        for layer_idx, smapler_fn in enumerate(sample_fns):
            ids = sampler_fns(ids=ids).contiguous().view(-1)
            tmp_feats = feats[ids] if has_feats else None
            all_feats.append(self.prep(ids, temp_feats, layer_idx=layer_idx +1))

        # Sequentially apply lyes, per original
        # Each iteration reduces length of aray by one
        for agg_laer in self.agg_layers.children():
            all_feats = [agg_layer()]




if __name__  == "__main__":
    args = parse_args()
    set.seeds(args.seed)

    # --
    # Load problem

    # TODO write NodeProblem
    problem = NodeProblem(problem_path = args.problem_path, cuda = args.cuda)


    # --
    # Define model
    n_train_samples = map(int, args.n_train_samples.split(','))
    n_val_samples = map(int, args.n_val_samples.split(','))
    output_dims = map(int, args.output_dims.split(','))
    model = GSSupervised(**{
        "sampler_class": sampler_lookup[args.sampler_class],
        "adj": problem.adj,
        "train_adj" : problem.train_adj,

        "prep_class": prep_lookup[args.prep_class],
        "aggregator_class" :aggregator_lookup[args.aggregtor_class],

        "input_dim" : problem.feats_dim,
        "n_nodes" : problem.n_nodes,
        "n_classes": problem.n_classes,
        "layer_specs": [
            {
                "n_train_samples" :n_train_samples[0],
                "n_val_samples" : n_val_samples[0],
                'output_dim' :output_dims[0],
                "activateion": F.relu,
            },
            {
                "n_train_samples" :n_train_samples[1],
                "n_val_samples" : n_val_samples[1],
                "output_dim" : output_dims[1],
                "activation" : lambda x: x,
            }
        ],
        "lr_init" : args.lr_init,
        "r_schedule" : args.lr_schedule,
        "weight_decay" : args.weight_decay,
    })

