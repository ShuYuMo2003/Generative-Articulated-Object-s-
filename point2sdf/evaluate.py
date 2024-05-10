from decoder import Decoder
from pointnet_encoder import SimplePointnet as Condition_Encoder
from encoder_latent import Encoder

encoder         = Encoder(z_dim=_.z_dim, c_dim=0, leaky=_.leaky).to(device) # unconditional
decoder         = Decoder(z_dim=_.z_dim, c_dim=0, leaky=_.leaky).to(device) # unconditional
