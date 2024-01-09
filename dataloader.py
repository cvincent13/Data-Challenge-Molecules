import os
import os.path as osp
import torch
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch.utils.data import Dataset as TorchDataset
import pandas as pd

def randow_walk_se(graph, walk_length):
    try:
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(0)
    except:
       adj = torch.ones((1,1)) # For graphs with only 1 node
    deg = adj.sum(dim=1)
    deg_inv = 1./deg
    deg_inv[deg_inv == float('inf')] = 0
    P = adj*deg_inv.view(-1,1)

    rwse = []
    Pk = P.clone().detach()
    for k in range(walk_length):
        rwse.append(torch.diagonal(Pk))
        Pk = Pk@P

    rwse = torch.stack(rwse, dim=-1)
    graph.rwse = rwse
    return graph

class AddRWStructEncoding(T.BaseTransform):
    def __init__(self, walk_length):
        self.walk_length = walk_length
    def __call__(self, graph):
        return randow_walk_se(graph, self.walk_length)


class GraphTextDataset(Dataset):
    def __init__(self, root, gt, split, tokenizer=None, graph_transform=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)   
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        
        self.graph_transform = graph_transform
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f: #get mol2vec features:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            text_input = self.tokenizer([self.description[1][cid]],
                                   return_tensors="pt", 
                                   truncation=True, 
                                   max_length=256,
                                   padding="max_length",
                                   add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])
            if self.graph_transform is not None:
               data = self.graph_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    

class GraphDatasetTrain(Dataset):
    def __init__(self, root, gt, split, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDatasetTrain, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
    
class GraphDataset(Dataset):
    def __init__(self, root, gt, split, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
