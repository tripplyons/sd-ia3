import torch
import torch.nn as nn


class IA3CrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.weight = nn.Parameter(torch.empty((hidden_size,)))
        self.bias = nn.Parameter(torch.empty((hidden_size,)))

        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

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

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # modulation
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states + \
            (self.weight * hidden_states + self.bias).to(original_dtype)

        return hidden_states


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
