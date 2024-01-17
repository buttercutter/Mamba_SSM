# [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange, repeat
from tqdm import tqdm

import math
import os
import urllib.request
from zipfile import ZipFile

from transformers import AutoTokenizer



torch.autograd.set_detect_anomaly(True)
debugging_is_on = 0

def print_tensor_info(tensor_name, tensor):
    # Check if tensor is floating point, and convert if necessary
    tensor_float = tensor.float() if not tensor.is_floating_point() else tensor

    # Gather the information
    info = {
        "shape": tuple(tensor.shape),
        "min/max": (tensor.min().item(), tensor.max().item()),
        "mean": tensor_float.mean().item(),
        "std": tensor_float.std().item()
    }

    # Print the default representation and the extra information
    print(f"{tensor_name} = {tensor}")
    for key, value in info.items():
        print(f"{key}: {value}")



USE_MAMBA = 1
USE_TRANSFORMER = ~USE_MAMBA
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# User hyperparameters
d_model = 16
state_size = 64  # Example state size
seq_len = 100  # Example sequence length
batch_size = 128  # Example batch size


class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        #self.A = nn.Parameter(torch.ones(d_model, state_size, device=device))
        #self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        #nn.init.xavier_uniform_(self.A)

        # S4D real initialization, MAMBA removed imaginary portions for S4D-Inv and S4D-Lin initialization schemes
        # described in [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893)
        # https://github.com/state-spaces/mamba/blob/fb7b5310fa865dbd62aa059b1e26f2b431363e2a/mamba_ssm/modules/mamba_simple.py#L103-L108C23
        A = repeat(
            torch.arange(1, state_size + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_model,
        ).contiguous()

        A_log = torch.log(A)  # For numerical stability during training process
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.A = torch.zeros_like(self.A_log)
        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        #self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        # Initialize delta parameter using a uniform distribution and apply the inverse softplus
        uniform_distribution = torch.distributions.Uniform(0.001, 0.1)
        # Sample from the uniform distribution and then apply the inverse softplus
        self.delta = self.inverse_softplus(uniform_distribution.sample((batch_size, self.seq_len, self.d_model)))

        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # h should have dimensions [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)


    def inverse_softplus(self, y):
        return torch.log(torch.exp(y) - 1)

    def discretization(self):
        # discretization function is defined based on the MAMBA paper's description using ZOH on page 28
        # in Section C : Mechanics on Selective SSMs
        # See also "Zero-order hold discretization" maths proof inside https://studywolf.wordpress.com/tag/zero-order-hold/
        """
        Here is an explanation of the mathematical rationale for the formulation of Δt used in Mamba:

        The key idea is that Δt controls the discretization rate of the continuous SSM dynamics. By making Δt input-dependent, it introduces selectivity into the discrete transition matrices.

        Specifically, in Mamba they parameterize Δt as:

        Δt = τΔ(Parameter + sΔ(xt))

        Where:

        - Parameter is a learned scalar parameter that controls the baseline discretization rate
        - sΔ(xt) is a projection that makes Δt input-dependent by computing a value based on xt
        - τΔ(x) = softplus(x) transforms the result to be positive through the softplus nonlinearity

        The rationale for this formulation is:
        - Parameter provides a reasonable default discretization rate
        - sΔ(xt) injects input-dependence through the projection
        - softplus ensures Δt is positive as required to be a valid timestep
        - The projection sΔ allows the model to learn to modulate Δt based on the input xt
        - This modulation creates selectivity in how rapidly or slowly the states update

        So in summary, the learned input-dependent projection allows Δt, and thus the discrete dynamics, to become selective. The softplus and scalar parameter provide useful inductive biases on top of this flexibility.

        The end result is discrete transition matrices that are selective on the input, enabling powerful sequence modeling capabilities.

        Credit: Claude2 AI chatbot
        """

        # For numerical stability during training process
        self.A = -torch.exp(self.A_log.float())  # (d_model, state_size)

        #print(f"self.A.shape = {self.A.shape}")
        #print(f"self.B.shape = {self.B.shape}")
        #print(f"self.delta.shape = {self.delta.shape}")

        # inverse() only supports square matrix
        #dB = torch.matmul(torch.inverse(A * delta), torch.matmul(dA - torch.eye(A.shape[0]), B))
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        # https://github.com/state-spaces/mamba/blob/0131c1e94a46fc9f70bcfc9d57962963bb2f0b9e/mamba_ssm/modules/mamba_simple.py#L240
        #dA = torch.matrix_exp(A * delta)  # matrix_exp() only supports square matrix
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        #print(f"self.dA.shape = {self.dA.shape}")
        #print(f"self.dA.requires_grad = {self.dA.requires_grad}")

        return self.dA, self.dB

    def forward(self, x):
        # Refer to Algorithm 2 in the MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)

        # "a large ∆ resets the state `h` and focuses on the current input `x`,
        # while a small ∆ persists the state and ignores the current input."
        self.delta = F.softplus(self.fc1(x))

        # Uses ZOH as in MAMBA, Hungry Hippo still uses bilinear transform for discretization
        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:  # this will trigger in-place runtime error if without using `h_new`
            #print(f"self.dA = {self.dA}, self.dB = {self.dB}")
            #print(f"self.dA.shape = {self.dA.shape}")
            #print(f"self.dB.shape = {self.dB.shape}")
            #print(f"x.shape = {x.shape}")
            #print(f"self.h.shape = {self.h.shape}")
            #print(f"self.C.shape = {self.C.shape}")

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                #print("Adjusting h_new for the different batch size of input data `x`")
                different_batch_size = True

                # Resize self.h to match the current batch size
                h_new =  torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x, "b l d -> b l d 1") * self.dB

            else:
                different_batch_size = False
                h_new =  torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y needs to have a shape of [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            # Update self.h with the detached state of h_new
            # Only do this if retaining gradients for self.h is not necessary for backprop
            # Otherwise, store h_new in a temporary list and update self.h after the loop
            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
            #print(f"temp_buffer.shape = {temp_buffer.shape}")

            #print(f"self.y = {self.y}")
            #print(f"self.dA.requires_grad = {self.dA.requires_grad}")
            #print(f"self.dB.requires_grad = {self.dB.requires_grad}")
            #print(f"self.C.requires_grad = {self.C.requires_grad}")
            #print(f"self.h.requires_grad = {self.h.requires_grad}")
            #print(f"self.y.requires_grad = {self.y.requires_grad}")

            return self.y

        else:  # this will not trigger in-place runtime error
            # h should have dimensions [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y needs to have a shape of [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, 2*d_model, device=device)
        self.out_proj = nn.Linear(2*d_model, d_model, device=device)

        # For residual skip connection
        self.D = nn.Linear(d_model, 2*d_model, device=device)

        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2*d_model, state_size, device)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

        # rmsnorm
        self.norm = RMSNorm(d_model, device=device)


    def forward(self, x, attention_mask=None):

        if attention_mask is not None:
            # Apply the attention mask
            x = x * attention_mask.unsqueeze(-1)

        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
        # Refer to Figure 3 in the MAMBA paper

        x = self.norm(x)

        x_proj = self.inp_proj(x)
        #print(f"x_proj.shape = {x_proj.shape}")

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)

        # Create a triangular mask of the same shape as the input sequence
        mask = torch.tril(torch.ones(seq_len, 2*d_model, device=device))

        # Add batch dimension with unsqueeze(0) -> (1, seq_len, seq_len)
        # Repeat batch dim to match x_conv batches with .repeat()
        current_batch_size = x.shape[0]
        mask = mask.repeat(current_batch_size, 1, 1)

        # Apply causal mask to zero out the masked regions
        x_conv = x_conv * mask
        #print(f"x_conv.shape = {x_conv.shape}")

        x_conv_act = F.silu(x_conv)  # Swish activation can be implemented as x * sigmoid(x)
        #print(f"x_conv_act.shape = {x_conv_act.shape}")

        x_ssm = self.S6(x_conv_act)
        #print(f"x_ssm.shape = {x_ssm.shape}")

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))
        #print(f"x_residual.shape = {x_residual.shape}")
        x_combined = x_ssm * x_residual
        #print(f"x_combined.shape = {x_combined.shape}")

        x_out = self.out_proj(x_combined)
        #print(f"x_out.shape = {x_out.shape}")

        return x_out


class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, vocab_size, device):
        super(Mamba, self).__init__()

        if vocab_size is None:
            vocab_size = d_model

        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)

        self.final_proj = nn.Linear(d_model, vocab_size, device=device)

    def forward(self, x, attention_mask=None):
        x = self.mamba_block1(x, attention_mask)
        x = self.mamba_block2(x, attention_mask)
        x = self.mamba_block3(x, attention_mask)

        x = self.final_proj(x)

        return x


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


# Example usage:
# Create a random input tensor
if USE_MAMBA:
    x = torch.rand(batch_size, seq_len, d_model, device=device)
    # Create the Mamba model
    mamba = Mamba(seq_len, d_model, state_size, None, device)

    # rmsnorm
    norm = RMSNorm(d_model)
    x = norm(x)

    # Forward pass
    test_output = mamba(x)
    print(f"test_output.shape = {test_output.shape}")  # Should be [batch_size, seq_len, d_model]


class Enwiki8Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['encoded_inputs'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        return item


# Define a function for padding
def pad_sequences_3d(sequences, max_len=None, pad_value=0):
    if sequences.ndim == 3:
        # Assuming sequences is a tensor of shape (batch_size, seq_len, feature_size)
        batch_size, seq_len, feature_size = sequences.shape

    else:
        # Assuming sequences is a tensor of shape (batch_size, seq_len)
        batch_size, seq_len = sequences.shape

    if max_len is None:
        max_len = seq_len + 1

    if sequences.ndim == 3:
        # Initialize padded_sequences with the pad_value
        padded_sequences = torch.full((batch_size, max_len, feature_size), fill_value=pad_value, dtype=sequences.dtype, device=sequences.device)
        # Pad each sequence to the max_len
        padded_sequences[:, :seq_len, :] = sequences

    else:
        # Initialize padded_sequences with the pad_value
        padded_sequences = torch.full((batch_size, max_len), fill_value=pad_value, dtype=sequences.dtype, device=sequences.device)
        # Pad each sequence to the max_len
        padded_sequences[:, :seq_len] = sequences

    return padded_sequences

def train(model, tokenizer, data_loader, optimizer, criterion, device, max_grad_norm=1.0, DEBUGGING_IS_ON=False):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        original_data = batch['input_ids'].clone().to(device)  # data without downsized dimension
        input_data = batch['encoded_inputs'].clone().to(device)  # data with downsized dimension for Mamba model
        attention_mask = batch['attention_mask'].clone().to(device)

        # In most sequence modeling tasks, like language modeling, the target should be the next token
        # in the sequence rather than the input token itself.
        # This is because the model's goal is to predict the next word given the previous words.
        # Shift the input data by one position to get the target, so that each target token
        # is the next token following the input token.
        target = original_data[:, 1:]
        input_data = input_data[:, :-1]

        #print("Before padding: ")
        #print(f"target.shape = {target.shape}")
        #print(f"input_data.shape = {input_data.shape}")

        # Pad all the sequences in the batch:
        input_data = pad_sequences_3d(input_data, pad_value=tokenizer.pad_token_id)
        target = pad_sequences_3d(target, max_len=original_data.size(1), pad_value=tokenizer.pad_token_id)

        #print("After padding: ")
        #print(f"target.shape = {target.shape}")
        #print(f"input_data.shape = {input_data.shape}")

        # For Mamba model, it can only accept downsized `input_data` due to RAM memory restriction
        # and already have a final_proj layer to upsize the `output` dimension to be the same as `target`
        output = model(input_data, attention_mask)
        #print(f"Output shape: {output.shape}")
        #print(f"Target shape: {target.shape}")

        loss = criterion(output.view(-1, vocab_size), target.view(-1))

        loss.backward(retain_graph=True)

        # Clip gradients: gradients are modified in place
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        for name, param in model.named_parameters():
           if 'out_proj.bias' not in name:
               # clip weights but not bias for out_proj
               torch.nn.utils.clip_grad_norm_(param, max_norm=max_grad_norm)

        if DEBUGGING_IS_ON:
            print("DEBUGGING IS ON !!!")
            print_tensor_info("output", output)
            print_tensor_info("target", target)

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    print(f"{name} gradient: {parameter.grad.data.norm(2)}")
                else:
                    print(f"{name} has no gradient")

        if USE_MAMBA and DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            # update self.h from temp_buffer
            #print(f"temp_buffer = {temp_buffer}")
            #print(f"temp_buffer.shape = {temp_buffer.shape}")
            #print(f"current_batch_size = {current_batch_size}")
            model.S6.h[:current_batch_size, ...].copy_(temp_buffer)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device, DEBUGGING_IS_ON=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            original_data = batch['input_ids'].clone().to(device)  # data without downsized dimension
            input_data = batch['encoded_inputs'].clone().detach().to(device)  # data with downsized dimension for Mamba model
            attention_mask = batch['attention_mask'].clone().detach().to(device)

            # In most sequence modeling tasks, like language modeling, the target should be the next token
            # in the sequence rather than the input token itself.
            # This is because the model's goal is to predict the next word given the previous words.
            # Shift the input data by one position to get the target, so that each target token
            # is the next token following the input token.
            target = original_data[:, 1:]
            input_data = input_data[:, :-1]

            #print("Before padding: ")
            #print(f"target.shape = {target.shape}")
            #print(f"input_data.shape = {input_data.shape}")

            # Pad all the sequences in the batch:
            input_data = pad_sequences_3d(input_data, pad_value=tokenizer.pad_token_id)
            target = pad_sequences_3d(target, max_len=original_data.size(1), pad_value=tokenizer.pad_token_id)

            #print("After padding: ")
            #print(f"target.shape = {target.shape}")
            #print(f"input_data.shape = {input_data.shape}")

            # For Mamba model, it can only accept downsized `input_data` due to RAM memory restriction
            # and already have a final_proj layer to upsize the `output` dimension to be the same as `target`
            output = model(input_data, attention_mask)
            #print(f"Output shape: {output.shape}")
            #print(f"Target shape: {target.shape}")

            loss = criterion(output.view(-1, vocab_size), target.view(-1))

            total_loss += loss.item()

            if DEBUGGING_IS_ON:
                print("DEBUGGING IS ON !!!")
                print_tensor_info("output", output)
                print_tensor_info("target", target)

    return total_loss / len(data_loader)

def calculate_perplexity(loss):
    return math.exp(loss)

def load_enwiki8_dataset():
    print(f"Download and extract enwiki8 data")
    url = "http://mattmahoney.net/dc/enwik8.zip"
    urllib.request.urlretrieve(url, "enwik8.zip")

    with ZipFile("enwik8.zip") as f:
        data = f.read("enwik8").decode("utf-8")

    return data

# Tokenize and encode the dataset
def encode_dataset(tokenizer, text_data):
    def batch_encode(tokenizer, text_data, batch_size=1000):
        # Tokenize in batches
        batched_input_ids = []
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            inputs = tokenizer(batch, add_special_tokens=True, truncation=True,
                               padding='max_length', max_length=seq_len,
                               return_tensors='pt')
            batched_input_ids.append(inputs['input_ids'])
        return torch.cat(batched_input_ids)

    # Assuming enwiki8_data is a list of sentences
    input_ids = batch_encode(tokenizer, enwiki8_data)

    # vocab_size is the number of unique tokens in the tokenizer's vocabulary
    global vocab_size
    vocab_size = len(tokenizer.vocab)  # Note that for some tokenizers, we might access the vocab directly
    print(f"vocab_size = {vocab_size}")

    # Create an embedding layer
    # embedding_dim is the size of the embedding vectors (MAMBA model's D)
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    # Pass `input_ids` through the embedding layer
    # This will change `input_ids` from shape [B, L] to [B, L, D]
    #encoded_inputs = embedding_layer(input_ids)   ## this eats memory, so use batched_embedding_calls instead
    def batch_embedding_calls(input_ids, embedding_layer, batch_size=256):
        # Check if input_ids is already a tensor, if not convert it
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Calculate the number of batches needed
        num_batches = math.ceil(input_ids.size(0) / batch_size)

        # List to hold the output embeddings
        output_embeddings = []

        # Process each batch
        for i in range(num_batches):
            # Calculate start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Get the batch
            input_id_batch = input_ids[start_idx:end_idx]

            # Call the embedding layer
            with torch.no_grad():  # No need gradients for this operation
                batch_embeddings = embedding_layer(input_id_batch)

            # Append the result to the list
            output_embeddings.append(batch_embeddings)

        # Concatenate the embeddings from each batch into a single tensor
        all_embeddings = torch.cat(output_embeddings, dim=0)

        return all_embeddings

    # `input_ids` is a list or tensor of the input IDs and `embedding_layer` is model's embedding layer
    if USE_MAMBA:
        # Set `batch_size` to a value that works for memory constraints
        # batch_embedding_calls() is very slow, not suitable to implement directly during forward pass
        encoded_inputs = batch_embedding_calls(input_ids, embedding_layer, batch_size=1).float()

    elif USE_TRANSFORMER:
        encoded_inputs = input_ids.long()  # Cast input_ids to long if necessary

    attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.dtype)
    #print(f"attention_mask.shape = {attention_mask.shape}")
    #print(f"encoded_inputs.shape = {encoded_inputs.shape}")

    return encoded_inputs, attention_mask, input_ids


# Load a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

# Use an existing special token as the padding token.
#tokenizer.pad_token = tokenizer.eos_token


# Assuming encoded_inputs is a preprocessed tensor of shape [num_samples, seq_len, d_model]
if USE_MAMBA:
    encoded_inputs_file = 'encoded_inputs_mamba.pt'
elif USE_TRANSFORMER:
    encoded_inputs_file = 'encoded_inputs_transformer.pt'

if os.path.exists(encoded_inputs_file):
    print("Loading pre-tokenized data...")
    encoded_inputs = torch.load(encoded_inputs_file)
else:
    print("Tokenizing raw data...")
    enwiki8_data = load_enwiki8_dataset()
    encoded_inputs, attention_mask, input_ids = encode_dataset(tokenizer, enwiki8_data)
    torch.save(encoded_inputs, encoded_inputs_file)
    print(f"finished tokenizing data")


# Combine into a single dictionary
data = {
    'input_ids': input_ids,
    'encoded_inputs': encoded_inputs,
    'attention_mask': attention_mask
}

# Split the data into train and validation sets
total_size = len(data['encoded_inputs'])
train_size = int(total_size * 0.8)

train_data = {key: val[:train_size] for key, val in data.items()}
val_data = {key: val[train_size:] for key, val in data.items()}

train_dataset = Enwiki8Dataset(train_data)
val_dataset = Enwiki8Dataset(val_data)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Initialize the model
if USE_MAMBA:
    model = Mamba(seq_len, d_model, state_size, vocab_size, device).to(device)

elif USE_TRANSFORMER:
    from transformers import AutoModel

    # Create TinyBert model instance
    bert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(device)

    print(f"bert_model.config.hidden_size = {bert_model.config.hidden_size}")

    class NextTokenPredictor(nn.Module):
        def __init__(self, bert_model, vocab_size):
            super(NextTokenPredictor, self).__init__()
            self.bert = bert_model
            self.predictor = nn.Linear(bert_model.config.hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            prediction_scores = self.predictor(sequence_output)
            return prediction_scores

    model = NextTokenPredictor(bert_model, vocab_size).to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-3)

# Training loop
num_epochs = 25  # Number of epochs to train for

for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
    train_loss = train(model, tokenizer, train_loader, optimizer, criterion, device, max_grad_norm=10.0, DEBUGGING_IS_ON=debugging_is_on)
    val_loss = evaluate(model, val_loader, criterion, device, DEBUGGING_IS_ON=debugging_is_on)
    val_perplexity = calculate_perplexity(val_loss)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}')

    if train_loss < 0 or val_loss < 0:
        debugging_is_on = 1
