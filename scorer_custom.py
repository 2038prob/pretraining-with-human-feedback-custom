# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
# from transformers import get_linear_schedule_with_warmup
# from datasets import load_dataset

# class CustomModel(nn.Module):
#     def __init__(self, pretrained_model_name='gpt2'):
#         super(CustomModel, self).__init__()
#         self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
#         # Adding a linear layer to map the hidden states to a single output
#         self.regressor = nn.Linear(self.gpt2.config.hidden_size, 1)

#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_states = outputs.last_hidden_state
#         # Using the representation of the first token ([CLS] token) for prediction
#         # Or you can use another approach to aggregate the sequence representation
#         pooled_output = last_hidden_states[:, 0]
#         return self.regressor(pooled_output)

# def custom_loss(outputs, labels):
#     mse_loss = torch.nn.MSELoss()
#     # Assume outputs are now directly comparable to labels
#     loss = mse_loss(outputs.squeeze(-1), labels.float())
#     return loss

# def train(model, tokenizer, dataset, device, epochs=1, batch_size=8, lr=5e-5):
#     model.to(device)
#     model.train()

#     optimizer = AdamW(model.parameters(), lr=lr)
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(epochs):
#         for batch in train_loader:
#             print(f'Batch : {batch}')
#             inputs = tokenizer(batch['tweet'], return_tensors='pt', padding=True, truncation=True, max_length=512)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             labels = batch['toxicity'].to(device) # Assuming you have a labels field in your dataset

#             optimizer.zero_grad()
#             outputs = model(**inputs)
#             loss = custom_loss(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             print(f"Epoch: {epoch}, Loss: {loss.item()}")


# # Example usage
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # model = GPT2LMHeadModel.from_pretrained('gpt2')
# model = CustomModel()
# tokenizer.pad_token = tokenizer.eos_token

# # Load or prepare your dataset
# dataset = load_dataset("csv", data_files="archive/FinalBalancedDataset.csv")['train']
# # Ensure your dataset has a 'text' and 'labels' field. You might need to preprocess your dataset accordingly.

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train(model, tokenizer, dataset, device)

###########################################################################

# from detoxify import Detoxify
# import torch
# import os

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PROMPTS = ['This is a cute cat',
#            'F***',
#            'Fuck',
#            'Fuck you',
#            'Fuck you you fucking fuck',
# ]

# print(f"Using {DEVICE}")

# detoxify_model = Detoxify('unbiased', device=DEVICE)

# scores = detoxify_model.predict(PROMPTS)

# for i, prompt in enumerate(PROMPTS):
#     print(f'Prompt : {prompt} \nScore : {scores["toxicity"][i]}\n')

###########################################################################

import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Parameters:
    - p: Tensor representing the first probability distribution.
    - q: Tensor representing the second probability distribution.
    
    Returns:
    - jsd: The Jensen-Shannon Divergence between p and q.
    """
    # Ensure distributions are normalized (sum to 1)
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate the midpoint distribution
    m = 0.5 * (p + q)
    
    # Compute KL Divergences between p, q and m
    kl_pm = torch.sum(p * torch.log(p / m), dim=-1)
    kl_qm = torch.sum(q * torch.log(q / m), dim=-1)
    
    # Calculate the Jensen-Shannon Divergence
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return jsd


# Define the two distributions
p = Normal(torch.tensor([0.0]), torch.tensor([1.0]))  # Mean = 0, StdDev = 1
q = Normal(torch.tensor([1.0]), torch.tensor([1.0]))  # Mean = 1, StdDev = 1.5

# Calculate KL Divergence (P || Q)
kl_pq = kl_divergence(p, q)
# jsd = jensen_shannon_divergence(p, q)

print(f"KL Divergence (P || Q): {kl_pq.item()}")
# print(f"Jensen-Shannon Divergence: {jsd.item()}")