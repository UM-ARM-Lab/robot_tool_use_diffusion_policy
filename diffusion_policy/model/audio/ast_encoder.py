from transformers import ASTModel, ASTConfig, ASTFeatureExtractor
import torch

def ast_encoder(audio_encoder_cfg):

    config = ASTConfig()
    config.max_length = audio_encoder_cfg.max_length
    config.num_mel_bins = audio_encoder_cfg.num_mel_bins
    audio_model = ASTModel(config)
    ast_feature_extractor = ASTFeatureExtractor(
                num_mel_bins=audio_encoder_cfg.num_mel_bins, 
                max_length=audio_encoder_cfg.max_length,
                do_normalize=False,
                # mean=audio_encoder_cfg.norm_spec.mean,
                # std=audio_encoder_cfg.norm_spec.std
            )
    feature_dim = 768





class ASTEncoder(nn.Module):
    def __init__(self, audio_encoder_cfg):
        super(ASTEncoder, self).__init__()
        self.audio_model = ast_encoder(audio_encoder_cfg)
        self.feature_dim = 768

    def forward(self, audio_input):

        audio = audio.reshape(B, audio.shape[-1]*T)
        audio = self.key_transform_map[key][0](audio)
        audio = self.ast_feature_extractor(audio.cpu().numpy(), sampling_rate=16000)['input_values']
        if self.audio_encoder_cfg.norm_spec.is_norm: # normalize to -1 and 1
            audio = (np.array(audio) - self.audio_encoder_cfg.norm_spec.min) \
                / (self.audio_encoder_cfg.norm_spec.max - self.audio_encoder_cfg.norm_spec.min)
            audio = audio * 2 - 1
        audio = torch.tensor(np.array(audio)).to(self.device)
        audio = self.key_transform_map[key][1:](audio)
        if self.audio_encoder_cfg.mask_robot:
            audio[:, :, :8] = 0


        raw_feature = self.key_model_map[key](audio, output_hidden_states=True)
        raw_feature = raw_feature.last_hidden_state
    

    return raw_feature
 