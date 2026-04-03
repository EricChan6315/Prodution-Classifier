import torch
import torch.nn as nn
from transformers import AutoModel

class MERT_AES(nn.Module):

    def __init__(self,
                proj_num_layer: int = 1,
                proj_ln: bool = False,
                proj_act_fn: str = "gelu",
                proj_dropout: float = 0,
                output_dim: int = 1,
                binary_classification: bool = False,
                freeze_encoder: bool = True,  # trf encoder freeze true means no weight update
                model_name: str = "m-a-p/MERT-v1-95M"
                ):
        super().__init__()
        self.proj_num_layer = proj_num_layer
        self.proj_ln = proj_ln
        self.proj_act_fn = proj_act_fn
        self.proj_dropout = proj_dropout
        self.output_dim = output_dim
        self.binary_classification = binary_classification
        self.freeze_encoder = freeze_encoder
        self.model_name = model_name
        
        # 1. Load the backbone (ensure output_hidden_states=True)
        self.mert = AutoModel.from_pretrained(self.model_name, output_hidden_states=True, trust_remote_code=True)
        self.num_layers = self.mert.config.num_hidden_layers + 1 # +1 for embedding layer
        self.hidden_size = self.mert.config.hidden_size
        
        # 2. Learnable weights for each layer (The "AES Secret Sauce")
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / (self.num_layers))

        # 3. Projection head for classification 
        self.proj_layer = nn.Sequential(
                *create_mlp_block(
                    self.hidden_size,
                    self.output_dim,
                    self.proj_num_layer,
                    self.proj_act_fn,
                    self.proj_ln,
                    self.binary_classification,
                    dropout=self.proj_dropout,
                )
            )
    def forward(self, input_values):
        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.mert(input_values)
        # Extract all hidden states: (num_layers, batch, seq_len, hidden_size)
        all_layers = torch.stack(outputs.hidden_states)
        
        # Apply learnable weights across layers
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted_sum = (all_layers * weights[:, None, None, None]).sum(0)
        
        # Mean Pooling over time (dim 1)
        embedding = weighted_sum.mean(dim=1)

        score = self.proj_layer(embedding)
        
        return score
    

def create_mlp_block(input_dim, output_dim, num_layer, act_fn, layer_norm, binary_classification, dropout=0):
    proj_layer = []
    for ii in range(num_layer):
        if ii == num_layer - 1:
            proj_layer.append(nn.Linear(input_dim, output_dim))
            if binary_classification:
                proj_layer.append(nn.Sigmoid())
        else:
            proj_layer.append(nn.Linear(input_dim, input_dim))
            if layer_norm:
                proj_layer.append(nn.LayerNorm(normalized_shape=(input_dim)))
            if act_fn == "gelu":
                proj_layer.append(nn.GELU())
            else:
                raise ValueError()
            if dropout != 0:
                proj_layer.append(nn.Dropout(p=dropout))
    return proj_layer