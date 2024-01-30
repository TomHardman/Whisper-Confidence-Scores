import torch
n_audio = 3
n_group = 5

audio_features = torch.rand((1, 1500, 768))
initial_tokens = (0, 0)
audio_features = audio_features.repeat_interleave(n_group, dim=0)
tokens = torch.tensor([initial_tokens]).repeat(n_audio, 1)
tokens = tokens.repeat_interleave(n_group, dim=0)
tokens += torch.tensor([[i//n_audio, i % n_audio] for i in range(15)]) # group, audio
tokens = tokens.reshape(n_audio, n_group, -1)
print('yes')