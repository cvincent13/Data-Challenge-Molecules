import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
import torch_geometric
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np

def randow_walk_se(graph, walk_length):
    try:
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(0)
    except:
       adj = torch.zeros((graph.num_nodes,graph.num_nodes)) # For graphs without connections between nodes
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
    def __init__(self, root, gt, split, tokenizer=None, nltk_tokenizer=None, word2idx=None, graph_transform=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.nltk_tokenizer = nltk_tokenizer
        self.word2idx = word2idx
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
            if self.tokenizer:
                text_input = self.tokenizer([self.description[1][cid]],
                                    return_tensors="pt", 
                                    truncation=True, 
                                    max_length=256,
                                    padding="max_length",
                                    add_special_tokens=True,)
                edge_index, x = self.process_graph(raw_path)
                data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])
                
            elif self.nltk_tokenizer:
                tokenized_text = self.nltk_tokenizer(self.description[1][cid])
                indexed_text = [self.word2idx.get(w, self.word2idx['UNK'])+1 for w in tokenized_text[:256]]
                input_ids = torch.zeros(256, dtype=torch.long)
                input_ids[:len(indexed_text)] = torch.LongTensor(indexed_text)
                edge_index, x = self.process_graph(raw_path)
                data = Data(x=x, edge_index=edge_index, input_ids=input_ids, attention_mask=torch.Tensor(0))
               
            else:
               edge_index, x = self.process_graph(raw_path)
               data = Data(x=x, edge_index=edge_index, text=self.description[1][cid])
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
    
    
class GraphDataset(Dataset):
    def __init__(self, root, gt, split, graph_transform=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None)
        self.cids = self.description[0].tolist()

        self.graph_transform = graph_transform
        
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
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer=None, nltk_tokenizer=None, word2idx=None, max_length=256):
        self.tokenizer = tokenizer
        self.nltk_tokenizer = nltk_tokenizer
        self.word2idx = word2idx
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

        if self.tokenizer:
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
        elif self.nltk_tokenizer:
            tokenized_text = self.nltk_tokenizer(sentence)
            indexed_text = [self.word2idx.get(w, self.word2idx['UNK'])+1 for w in tokenized_text[:256]]
            input_ids = torch.zeros(256, dtype=torch.long)
            input_ids[:len(indexed_text)] = torch.LongTensor(indexed_text)
            return {
                'input_ids': input_ids,
                'attention_mask': torch.Tensor(0)
            }


def drop_node_augment(graph, p, mask):
    _, _, node_mask = torch_geometric.utils.dropout_node(graph.edge_index, p, num_nodes=graph.num_nodes)
    node_idx = torch_geometric.utils.mask_to_index(node_mask)
    aug_graph = graph.subgraph(node_idx)
    return aug_graph

def edge_pert_augment(graph, p, mask):
    aug_graph = graph.clone()
    dropped_edge_index, _ = torch_geometric.utils.dropout_edge(graph.edge_index, p, force_undirected=True)
    perturbed_edge_index, _ = torch_geometric.utils.add_random_edge(dropped_edge_index, p, force_undirected=True)
    aug_graph.edge_index = perturbed_edge_index
    return aug_graph

def attr_mask_augment(graph, p, mask):
    n = graph.num_nodes
    masked_nodes = torch.randperm(n)[:int(np.round(p*n))]
    aug_graph = graph.clone()
    for node in masked_nodes:
        aug_graph.x[node] = torch.FloatTensor(mask)
    return aug_graph

def find_neighbors(edge_index, node_index):
    try:
        mask = (edge_index[0] == node_index) | (edge_index[1] == node_index)
        neighbors = edge_index[:, mask]
        neighbors = neighbors[neighbors != node_index].unique()
        neighbors = [n.item() for n in neighbors]
    except:
       neighbors = []
    return neighbors

def subgraph_augment(graph, p, mask):
    n = graph.num_nodes
    subgraph_num_nodes = np.round(n*(1-p))
    start_node = np.random.randint(n)
    sampled_nodes = [start_node]
    neighbors_nodes = find_neighbors(graph.edge_index, start_node)
    while len(sampled_nodes) < subgraph_num_nodes:
        possible_nodes = [n for n in neighbors_nodes if n not in sampled_nodes]
        # Multiple connected components: need to sample from another connected component
        if len(possible_nodes) == 0:
            new_node = np.random.randint(n)
            while new_node in sampled_nodes:
                new_node = np.random.randint(n)
            sampled_nodes.append(new_node)
            neighbors_nodes += find_neighbors(graph.edge_index, new_node)

        else:
            sample = np.random.randint(len(neighbors_nodes))
            new_node = neighbors_nodes[sample]
            while new_node in sampled_nodes:
                sample = np.random.randint(len(neighbors_nodes))
                new_node = neighbors_nodes[sample]
            sampled_nodes.append(new_node)
            neighbors_nodes += find_neighbors(graph.edge_index, new_node)

    try:
        aug_graph = graph.subgraph(torch.LongTensor(sampled_nodes))
    except:
       # If graph has no edges
       aug_graph = torch_geometric.data.Data()
       sampled_nodes_mask = torch_geometric.utils.index_to_mask(torch.LongTensor(sampled_nodes), size=n)
       aug_graph.x = graph.x[sampled_nodes_mask]
       aug_graph.edge_index = graph.edge_index
       aug_graph.rwse = graph.rwse[sampled_nodes_mask]

    return aug_graph

class GraphDatasetPretrain(Dataset):
    def __init__(self, root, gt, split, graph_augment1, graph_augment2, aug_p, transform=None, pre_transform=None, graph_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)
        self.cids = self.description[0].tolist()

        self.graph_augment1 = graph_augment1
        self.graph_augment2 = graph_augment2
        self.aug_p = aug_p
        self.graph_transform = graph_transform
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDatasetPretrain, self).__init__(root, transform, pre_transform)

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
        return osp.join(self.root, 'pretrain/', self.split)

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

            if self.graph_transform is not None:
               data = self.graph_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        mask = self.gt['UNK']
        aug_data1 = self.graph_augment1(data, self.aug_p, mask)
        aug_data2 = self.graph_augment2(data, self.aug_p, mask)
        return aug_data1, aug_data2

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        mask = self.gt['UNK']
        aug_data1 = self.graph_augment1(data, self.aug_p, mask)
        aug_data2 = self.graph_augment2(data, self.aug_p, mask)
        return aug_data1, aug_data2
    
    def get_idx_to_cid(self):
        return self.idx_to_cid