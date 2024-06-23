import torch.nn as nn
import torch
from transformers import PreTrainedModel, LlamaConfig, LlamaModel

class LlamaForMaskedLLM(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(config.hidden_size, 1)  # Regression head to predict floating-point values
        self.init_weights()
        self.tokenizer = tokenizer
    
    def resize_token_embeddings(self, new_num_tokens):
        self.model.resize_token_embeddings(new_num_tokens)
        self.regression_head = nn.Linear(self.config.hidden_size, 1).to(self.model.device)
        self.config.vocab_size = new_num_tokens

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings
        
    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable()
        self.supports_gradient_checkpointing = True 
    
                
    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        if labels is not None:
            labels = labels.to(self.model.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        
        regression_scores = self.regression_head(sequence_output).squeeze(-1)  # Regression predictions

        regression_loss = None
        if labels is not None: 
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
            if mask_positions[0].size(0) > 0:
                masked_regression_scores = regression_scores[mask_positions]
                masked_regression_labels =labels[mask_positions[0]]            
                regression_loss = nn.MSELoss()(masked_regression_scores, masked_regression_labels)
                regression_loss = regression_loss.mean()
            else:
                regression_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                
        return (regression_loss, regression_scores) if regression_loss is not None else regression_scores
