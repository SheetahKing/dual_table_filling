import torch
from transformers import BertModel, BertPreTrainedModel, AutoModel
from transformers import BertTokenizer
from .table import TableEncoder
from .matching_layer import MatchingLayer
from .se_classifier import InferenceLayer
from .gcn.gcn_model import GCN
from .gcn.Adj import adj, ADJ
from .aline import aline_seq

class DTFModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.ADJ = ADJ()    
        self.gcn = GCN()

        self.table_encoder = TableEncoder(config)


        self.inference = InferenceLayer(config)

        self.matching = MatchingLayer(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask,
                sentences,  
                ids,
                start_label_masks, end_label_masks,
                t_start_labels=None, t_end_labels=None, 
                o_start_labels=None, o_end_labels=None, 
                table_labels_S=None, table_labels_E=None,
                polarity_labels=None, pairs_true=None):

        seq = self.bert(input_ids, attention_mask)[0]       

        table = self.table_encoder(seq, attention_mask)  
        output = self.inference(table, attention_mask, table_labels_S, table_labels_E)

        output['ids'] = ids
        output = self.matching(output, table, pairs_true, seq)
        return output









