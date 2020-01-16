import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loopyBP_factorGraph import FactorGraphMsgPassingLayer_NoDoubleCounting



class lbp_message_passing_network(nn.Module):
    def __init__(self, max_factor_state_dimensions):
        super(SetTransformer, self).__init__()
        self.layer1 = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**factor_graph.state_dimensions)
        self.layer2 = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**factor_graph.state_dimensions)
        self.layer3 = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**factor_graph.state_dimensions)
        self.layer4 = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**factor_graph.state_dimensions)


    def forward(self, factor_graph):
        layer1(factor_graph)
        layer2(factor_graph)
        layer3(factor_graph)
        layer4(factor_graph)
        