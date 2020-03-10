from transformers.modeling_bert import BertForQuestionAnswering, BertLayer
from transformers.modeling_albert import AlbertForQuestionAnswering
from transformers.modeling_distilbert import DistilBertForQuestionAnswering, TransformerBlock
from transformers.modeling_roberta import RobertaForQuestionAnswering
import copy

class BFQA(BertForQuestionAnswering):
    def __init__(self, config):
        super(BFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.bert.encoder.layer[sourceidx]
        if not link:
            mlp = BertLayer(self.config)
        self.bert.encoder.layer.insert(targetidx, mlp)
        oldct = self.config.num_hidden_layers
        newct = oldct + 1
        self.config.num_hidden_layers = newct #self
        self.bert.config.num_hidden_layers = newct #bertmodel
        #encoder does not seem to have config

class AFQA(AlbertForQuestionAnswering):
    def __init__(self, config):
        super(AFQA, self).__init__(config)

    def incLayers(self, nlayers):
        oldct = self.config.num_hidden_layers
        newct = oldct + nlayers
        self.config.num_hidden_layers = newct #self
        self.albert.config.num_hidden_layers = newct #albertmodel
        self.albert.encoder.config.num_hidden_layers = newct #encoder

class RFQA(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(RFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.roberta.encoder.layer[sourceidx]
        if not link:
            mlp = BertLayer(self.config)
        self.bert.encoder.layer.insert(targetidx, mlp)
        oldct = self.config.num_hidden_layers
        newct = oldct + 1
        self.config.num_hidden_layers = newct #self
        self.bert.config.num_hidden_layers = newct #bertmodel
        #encoder does not seem to have config

class DFQA(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(DFQA, self).__init__(config)

    def dupeLayer(self, sourceidx, targetidx, link = False):
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if link:
            mlp = self.distilbert.transformer.layer[sourceidx]
        if not link:
            mlp = TransformerBlock(self.config)
        self.distilbert.transformer.layer.insert(targetidx, mlp)
        oldct = self.config.n_layers
        newct = oldct + 1
        self.config.n_layers = newct #self
        self.bert.config.n_layers = newct #bertmodel
        #encoder does not seem to have config



