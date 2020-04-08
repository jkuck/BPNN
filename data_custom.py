from torch_geometric.data import Data

import re
import copy
import warnings

import torch
import torch_geometric
from torch_sparse import coalesce
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

# from ..utils.num_nodes import maybe_num_nodes

__num_nodes_warn_msg__ = (
    'The number of nodes in your data object can only be inferred by its {} '
    'indices, and hence may result in unexpected batch-wise behavior, e.g., '
    'in case there exists isolated nodes. Please consider explicitly setting '
    'the number of nodes for this data object by assigning it to '
    'data.num_nodes.')

#this class is only a slight modification from pytorch geometric 'Data' class.
#the difference is it implements num_vars and num_factors (based on num_nodes)
#it doesn't require all input arguments when being contstructed, compared to DataFactorGraph.
class DataFactorGraph_partial(Data):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        norm (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)
    The data object is not restricted to these attributes and can be extented
    by any other additional data.
    Example::
        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.norm = norm
        self.face = face
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if torch_geometric.is_debug_enabled():
            self.debug()


    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'norm', 'batch'):
            return item.size(self.__cat_dim__(key, item))
#         if self.face is not None:
#             warnings.warn(__num_nodes_warn_msg__.format('face'))
#             return maybe_num_nodes(self.face)
#         if self.edge_index is not None:
#             warnings.warn(__num_nodes_warn_msg__.format('edge'))
#             return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    #jdk
    @property
    def num_vars(self):
        r"""Returns or sets the number of variables in the factor graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_vars = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, 'prv_factor_beliefs'):
            return self.prv_var_beliefs.size(self.__cat_dim__('prv_factor_beliefs', self.prv_var_beliefs))
        else:
            return None

    @num_vars.setter
    def num_vars(self, num_vars):
        self.__num_vars__ = num_vars     
        
        
    #jdk
    @property
    def num_factors(self):
        r"""Returns or sets the number of factors in the factor graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_factors = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, 'prv_factor_beliefs'):
            return self.prv_factor_beliefs.size(self.__cat_dim__('prv_factor_beliefs', self.prv_factor_beliefs))
        else:
            return None

    @num_factors.setter
    def num_factors(self, num_factors):
        self.__num_factors__ = num_factors            

        
