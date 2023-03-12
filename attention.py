import torch
import torch.nn as nn


# based on LoRACrossAttnProcessor
class IA3CrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.key_vector = nn.Parameter(torch.empty((hidden_size,)))
        self.value_vector = nn.Parameter(torch.empty((hidden_size,)))

        # start with zeros
        # this will start with no change to the base model
        nn.init.zeros_(self.key_vector)
        nn.init.zeros_(self.value_vector)

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # (IA)^3 changes
        original_dtype = key.dtype
        key = key + (self.key_vector * key).to(original_dtype)
        value = value + (self.value_vector * value).to(original_dtype)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# save to file
def save_attn_processors(unet, device, dtype, save_path):
    attn_processors = unet.attn_processors
    keys = list(attn_processors.keys())
    weights_dict = {}
    parameters_dict = {}

    for key in keys:
        processor = attn_processors[key].to(device).to(dtype)
        weights_dict[key] = processor.state_dict()
        parameters_dict[key] = {
            'hidden_size': processor.hidden_size
        }

    output_dict = {
        'weights': weights_dict,
        'parameters': parameters_dict
    }

    torch.save(output_dict, save_path)


# load from file
def load_attn_processors(unet, device, dtype, save_path):
    input_dict = torch.load(save_path)
    weights_dict = input_dict['weights']
    parameters_dict = input_dict['parameters']

    keys = list(weights_dict.keys())

    attn_processors = {}

    for key in keys:
        attn_processors[key] = IA3CrossAttnProcessor(
            hidden_size=parameters_dict[key]['hidden_size']
        ).to(device).to(dtype)
        attn_processors[key].load_state_dict(weights_dict[key])

    unet.set_attn_processor(attn_processors)
