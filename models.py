from transformers.modeling_bert import BertForQuestionAnswering, BertLayer
from transformers.modeling_albert import AlbertForQuestionAnswering
from transformers.modeling_distilbert import DistilBertForQuestionAnswering, TransformerBlock
from transformers.modeling_roberta import RobertaForQuestionAnswering
from transformers.modeling_xlnet import XLNetForQuestionAnswering, XLNetLayer
from transformers.modeling_xlm import XLMForQuestionAnswering, MultiHeadAttention, TransformerFFN
from torch import nn

import copy

class BFQA(BertForQuestionAnswering):
    def __init__(self, config):
        super(BFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.bert.encoder.layer[sourceidx]
        if not link:
            # mlp = BertLayer(self.config)
            mlp = copy.deepcopy(self.bert.encoder.layer[sourceidx])
        self.bert.encoder.layer.insert(targetidx, mlp)
        oldct = self.config.num_hidden_layers
        newct = oldct + 1
        self.config.num_hidden_layers = newct #self
        # self.bert.config.num_hidden_layers = newct #bertmodel, commented out since linked
        #encoder does not seem to have config
    def getLayers(self):
        return self.config.num_hidden_layers
#bert-base-uncased - 12 layers

class AFQA(AlbertForQuestionAnswering):
    def __init__(self, config):
        super(AFQA, self).__init__(config)

    def incLayers(self, nlayers):
        oldct = self.config.num_hidden_layers
        newct = oldct + nlayers
        self.config.num_hidden_layers = newct #self
        # self.albert.config.num_hidden_layers = newct #albertmodel
        # self.albert.encoder.config.num_hidden_layers = newct #encoder
    def getLayers(self):
        return self.config.num_hidden_layers
    def setLayers(self, nlayers):
        self.config.num_hidden_layers = nlayers
#albert-base-v2 - 12 layers


class RFQA(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(RFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.roberta.encoder.layer[sourceidx]
        if not link:
            mlp = copy.deepcopy(self.roberta.encoder.layer[sourceidx])
            # mlp = BertLayer(self.config)
        self.roberta.encoder.layer.insert(targetidx, mlp)
        oldct = self.config.num_hidden_layers
        newct = oldct + 1
        self.config.num_hidden_layers = newct #self
        # self.roberta.config.num_hidden_layers = newct #bertmodel
        #encoder does not seem to have config
    def getLayers(self):
        return self.config.num_hidden_layers
#roberta-base - 12 layer
#distilroberta-base - 6 layer

class DFQA(DistilBertForQuestionAnswering):
    def __init__(self, config):
        super(DFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.distilbert.transformer.layer[sourceidx]
        if not link:
            mlp = copy.deepcopy(self.distilbert.transformer.layer[sourceidx])
            # mlp = TransformerBlock(self.config)
        self.distilbert.transformer.layer.insert(targetidx, mlp)
        oldct = self.config.n_layers
        newct = oldct + 1
        self.config.n_layers = newct #self
        # self.distilbert.config.n_layers = newct #bertmodel
        #encoder does not seem to have config
    def getLayers(self):
        return self.config.n_layers
#distilbert-base-uncased - 6 layers

class XLNFQA(XLNetForQuestionAnswering):
    def __init__(self, config):
        super(XLNFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.transformer.layer[sourceidx]
        if not link:
            mlp = copy.deepcopy(self.transformer.layer[sourceidx])
            # mlp = XLNetLayer(self.config)
        self.transformer.layer.insert(targetidx, mlp)
        oldct = self.config.n_layer
        newct = oldct + 1
        self.config.n_layer = newct #self
        # self.transformer.config.n_layers = newct #bertmodel
        #encoder does not seem to have config
    def getLayers(self):
        return self.config.n_layer
#xlnet-base-cased

##TODO: XLMForQuestionAnswering
class XLMFQA(XLMForQuestionAnswering):
    def __init__(self, config):
        super(XLMFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            att = self.transformer.attentions[sourceidx]
            ln1 = self.transformer.layer_norm1[sourceidx]
            ff = self.transformer.ffns[sourceidx]
            ln2 = self.transformer.layer_norm2[sourceidx]
        if not link:
            att = copy.deepcopy(self.transformer.attentions[sourceidx])
            ln1 = copy.deepcopy(self.transformer.layer_norm1[sourceidx])
            ff = copy.deepcopy(self.transformer.ffns[sourceidx])
            ln2 = copy.deepcopy(self.transformer.layer_norm2[sourceidx])
            # att = MultiHeadAttention(self.transformer.n_heads, self.transformer.dim, config=self.config)
            # ln1 = nn.LayerNorm(self.transformer.dim, eps=config.layer_norm_eps)
            # ff = TransformerFFN(self.transformer.dim, self.transformer.hidden_dim, self.transformer.dim, config=self.config)
            # ln2 = nn.LayerNorm(self.transformer.dim, eps=config.layer_norm_eps)
        self.transformer.attentions.insert(targetidx, att)
        self.transformer.layer_norm1.insert(targetidx, ln1)
        self.transformer.ffns.insert(targetidx, ff)
        self.transformer.layer_norm2.insert(targetidx, ln2)
        oldct = self.config.n_layers
        newct = oldct + 1
        self.config.n_layers = newct #self
        # self.transformer.config.n_layers = newct #bertmodel
        #encoder does not seem to have config
    def getLayers(self):
        return self.config.n_layers
##Must duplicate self.transformer.attentions/layer_norm1/ffns/layer_norm2
##use n_layers
##xlm-mlm-en-2048 - 12 layers




