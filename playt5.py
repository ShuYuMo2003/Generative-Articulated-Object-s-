from transformer.model.encoder import T5Encoder
from rich import print

encoder = T5Encoder("google-t5/t5-large", "cache")

res = encoder(['I love something, like you.', 'I hate something', 'I have the world.']).to('cuda')

print(type(res), res.shape)