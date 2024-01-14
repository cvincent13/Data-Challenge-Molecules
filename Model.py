from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

from torch_geometric.nn.models import GraphSAGE, GIN
from torch_geometric.nn import GENConv, DeepGCNLayer, GCNConv, ResGatedGraphConv, GINEConv, Linear, global_mean_pool
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch
import torch


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, graph_hidden_channels, graph_layers):
        super(GraphEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_node_features, graph_hidden_channels))
        for i in range(1,graph_layers):
            self.conv_layers.append(GCNConv(graph_hidden_channels, graph_hidden_channels))

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = x.relu()
        x = global_mean_pool(x, batch)
        return x
    

class GraphEncoderSAGE(nn.Module):
    def __init__(self, num_node_features, graph_hidden_channels, graph_layers):
        super(GraphEncoderSAGE, self).__init__()
        self.gcn = GraphSAGE(num_node_features, graph_hidden_channels, graph_layers, graph_hidden_channels)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        x = self.gcn(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


class DeeperGCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers):
        super().__init__()


        self.layers = nn.ModuleList()
        conv = GENConv(num_node_features, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
        norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
        act = nn.ReLU(inplace=True)
        layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=1)
        self.layers.append(layer)

        for i in range(2, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

    def forward(self, x, edge_index):
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return x
    

class GraphEncoderDeep(nn.Module):
    def __init__(self, num_node_features, graph_hidden_channels, graph_layers):
        super(GraphEncoderDeep, self).__init__()
        self.gcn = DeeperGCN(num_node_features, graph_hidden_channels, graph_layers)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        x = self.gcn(x, edge_index)
        x = global_mean_pool(x, batch)
        return x
    

class GraphEncoderGIN(nn.Module):
    def __init__(self, num_node_features, graph_hidden_channels, graph_layers):
        super(GraphEncoderGIN, self).__init__()
        self.gcn = GIN(num_node_features, graph_hidden_channels, graph_layers)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        x = self.gcn(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class RWSEEncoder(nn.Module):
    def __init__(self, num_node_features, n_hidden, walk_length, dim_se, input_dropout):
        super(RWSEEncoder, self).__init__()
        self.embedding_x = nn.Linear(num_node_features, n_hidden-dim_se)
        self.norm = nn.BatchNorm1d(walk_length)
        self.embedding_se = nn.Linear(walk_length, dim_se)
        self.in_dropout = nn.Dropout(input_dropout)

    def forward(self, batch):
        rwse = batch.rwse
        rwse = self.norm(rwse)
        rwse = self.embedding_se(rwse)
        x = self.embedding_x(batch.x)
        x = torch.cat((x, rwse), 1)
        batch.x = self.in_dropout(x)
        return batch

class GPSLayer(nn.Module):
    def __init__(self, n_hidden, n_head, n_feedforward, dropout, attention_dropout, conv_type):
        super(GPSLayer, self).__init__()

        self.conv_type = conv_type
        if conv_type == 'GCN':
            self.conv = GCNConv(n_hidden, n_hidden)
        elif conv_type == 'Gated':
            self.conv = ResGatedGraphConv(n_hidden, n_hidden)
        elif conv_type == 'GINE':
            gin_nn = nn.Sequential(Linear(n_hidden, n_hidden),
                                   nn.ReLU(),
                                   Linear(n_hidden, n_hidden))
            self.conv = GINEConv(gin_nn)

        self.conv_norm = LayerNorm(n_hidden)
        self.conv_dropout = nn.Dropout(dropout)

        # Multi-head Attention
        self.attention_norm = LayerNorm(n_hidden)
        self.multihead_attention = nn.MultiheadAttention(n_hidden, n_head, attention_dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.feedforward_norm = LayerNorm(n_hidden)
        self.feedforward = nn.Sequential(nn.Linear(n_hidden, n_feedforward),
                                         nn.GELU(),
                                         nn.Linear(n_feedforward, n_hidden))
        self.feedforward_dropout = nn.Dropout(dropout)


    def forward(self, batch):
        x = batch.x
        x_res1 = x

        # GCN
        x_gcn = self.conv(x, batch.edge_index)
        x_gcn = self.conv_dropout(x_gcn)
        x_gcn = x_gcn + x_res1
        x_gcn = self.conv_norm(x_gcn, batch.batch)

        # Self-attention
        x_att, mask = to_dense_batch(x, batch.batch)
        x_att = self.multihead_attention(x_att, x_att, x_att, key_padding_mask=~mask, need_weights=False)[0][mask]
        x_att = self.attention_dropout(x_att)
        x_att = x_att + x_res1
        x_att = self.attention_norm(x_att)

        x = x_gcn + x_att

        # Feed-forward network
        x_res2 = x
        x = self.feedforward(x)
        x = self.feedforward_dropout(x)
        x = x + x_res2
        x = self.feedforward_norm(x)

        batch.x = x

        return batch
    

class GraphEncoderGPS(nn.Module):
    def __init__(self, 
                 num_node_features,  
                 graph_hidden_channels, 
                 graph_layers, 
                 n_head, 
                 n_feedforward, 
                 input_dropout, 
                 dropout, 
                 attention_dropout, 
                 conv_type,
                 walk_length,
                 dim_se):
        
        super(GraphEncoderGPS, self).__init__()
        self.node_encoder = RWSEEncoder(num_node_features, graph_hidden_channels, walk_length, dim_se, input_dropout)
        self.gps_layers = nn.ModuleList(
            [GPSLayer(graph_hidden_channels, n_head, n_feedforward, dropout, attention_dropout, conv_type) for _ in range(graph_layers)]
            )


    def forward(self, batch):
        batch = self.node_encoder(batch)

        for layer in self.gps_layers:
            batch = layer(batch)

        x = global_mean_pool(batch.x, batch.batch)
        return x

    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, graph_model_name, num_node_features, nout, nhid, graph_hidden_channels, graph_config):
        super(Model, self).__init__()
        graph_model_name = graph_model_name.lower()
        if graph_model_name == 'base':
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoder(num_node_features, graph_hidden_channels, graph_layers)
        elif graph_model_name == 'sage':
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoderSAGE(num_node_features, graph_hidden_channels, graph_layers)
        elif graph_model_name == 'deep':
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoderDeep(num_node_features, graph_hidden_channels, graph_layers)
        elif graph_model_name == 'gin':
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoderGIN(num_node_features, graph_hidden_channels, graph_layers)
        elif graph_model_name == 'gps':
            graph_layers = graph_config['graph_layer']
            n_head = graph_config['n_head']
            n_feedforward = graph_config['n_feedforward']
            input_dropout = graph_config['input_dropout']
            dropout = graph_config['dropout']
            attention_dropout = graph_config['attention_dropout']
            conv_type = graph_config['conv_type']
            walk_length = graph_config['walk_length']
            dim_se = graph_config['dim_se']
            self.graph_base = GraphEncoderGPS(num_node_features,
                                                graph_hidden_channels, 
                                                graph_layers, 
                                                n_head, 
                                                n_feedforward, 
                                                input_dropout, 
                                                dropout, 
                                                attention_dropout, 
                                                conv_type,
                                                walk_length,
                                                dim_se)
            
        self.projection_head = nn.Sequential(nn.Linear(graph_hidden_channels, nhid),
                                              nn.ReLU(),
                                              nn.Linear(nhid, nout))
        self.graph_encoder = nn.Sequential(self.graph_base, self.projection_head)

        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    


class GraphCL(nn.Module):
    def __init__(self, graph_config, nhid, nout):
        
        super(GraphCL, self).__init__()
        graph_model_name = graph_config['graph_model_name']
        graph_model_name = graph_model_name.lower()

        if graph_model_name == 'base':
            num_node_features = graph_config['num_node_features']
            graph_hidden_channels = graph_config['graph_hidden_channels']
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoder(num_node_features, graph_hidden_channels, graph_layers)

        elif graph_model_name == 'sage':
            num_node_features = graph_config['num_node_features']
            graph_hidden_channels = graph_config['graph_hidden_channels']
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoderSAGE(num_node_features, graph_hidden_channels, graph_layers)

        elif graph_model_name == 'deep':
            num_node_features = graph_config['num_node_features']
            graph_hidden_channels = graph_config['graph_hidden_channels']
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoderDeep(num_node_features, graph_hidden_channels, graph_layers)

        elif graph_model_name == 'gin':
            num_node_features = graph_config['num_node_features']
            graph_hidden_channels = graph_config['graph_hidden_channels']
            graph_layers = graph_config['graph_layer']
            self.graph_base = GraphEncoderGIN(num_node_features, graph_hidden_channels, graph_layers)

        elif graph_model_name == 'gps':
            num_node_features = graph_config['num_node_features']
            graph_hidden_channels = graph_config['graph_hidden_channels']
            graph_layers = graph_config['graph_layer']
            n_head = graph_config['n_head']
            n_feedforward = graph_config['n_feedforward']
            input_dropout = graph_config['input_dropout']
            dropout = graph_config['dropout']
            attention_dropout = graph_config['attention_dropout']
            conv_type = graph_config['conv_type']
            walk_length = graph_config['walk_length']
            dim_se = graph_config['dim_se']
            self.graph_base = GraphEncoderGPS(num_node_features,
                                                graph_hidden_channels, 
                                                graph_layers, 
                                                n_head, 
                                                n_feedforward, 
                                                input_dropout, 
                                                dropout, 
                                                attention_dropout, 
                                                conv_type,
                                                walk_length,
                                                dim_se)
            
        self.projection_head = nn.Sequential(nn.Linear(graph_hidden_channels, nhid),
                                              nn.ReLU(),
                                              nn.Linear(nhid, nout))


    def forward(self, batch):
        x = self.graph_base(batch)
        x = self.projection_head(x)
        return x
