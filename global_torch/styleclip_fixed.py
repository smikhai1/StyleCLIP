import clip
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from tqdm import tqdm

from training.networks import StyleVector
from manipulate import LoadModel
from clip_templates import imagenet_templates


# TODO: debug relevance scores estimation
# TODO: add support of saving and loading estimated channel relevances
# TODO: debug attribute manipulation
# TODO: make notebook with demo
from PIL import Image
import numpy as np

@torch.no_grad()
def encode_text_clip(text, model, device):
    augm_text = [tmp.format(text) for tmp in imagenet_templates]
    tokens = clip.tokenize(augm_text).to(device=device)
    embedds = model.encode_text(tokens)
    embedds /= embedds.norm(dim=-1, keepdim=True)
    embedds = embedds.mean(dim=0)
    return embedds


@torch.no_grad()
def encode_image_clip(image, model, device):
    """
    Args:
        image: torch.Tensor of shape [N, C, H, W] in [0, 1] range
        model: CLIP model object

    Returns:
        embedds: torch.Tensor of shape [N, d], where d is the CLIP's latent space dimension
    """
    image = image.to(device=device)
    image = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=True)

    image = TVF.normalize(image, mean=[0.48145466, 0.4578275, 0.40821073],
                          std=[0.26862954, 0.26130258, 0.27577711])
    embedds = model.encode_image(image)
    embedds /= embedds.norm(dim=-1, keepdim=True)
    return embedds


class StyleClipGlobal:
    def __init__(self, generator, clip_model, neutral_text, target_text,
                 include_torgb_layers=False, num_samples=100, device='cuda', seed=1112):
        torch.random.manual_seed(seed)
        self.generator = generator
        self.include_torgb_layers = include_torgb_layers
        self.device = device
        self.num_samples = num_samples
        self.clip_model = clip_model

        self.delta_t = None  # [d,]
        self.neutral_text = neutral_text
        self.target_text = target_text
        self._compute_clip_textual_direction()

        self.style_mean, self.style_std = None, None
        self._compute_mean_std_style()

        self.attribute_relevance_scores = None  # [S,]
        self._esitmate_relevance_scores()

    def _compute_clip_textual_direction(self):
        neutral_embedds = encode_text_clip(self.neutral_text, self.clip_model, self.device)
        target_embedds = encode_text_clip(self.target_text, self.clip_model, self.device)

        delta = target_embedds - neutral_embedds
        delta /= delta.norm(dim=-1, keepdim=True)
        self.delta_t = delta.squeeze()

    def _predict_image_embeddings(self, style_tensor, style_handler, styles_dict, alpha, channel_id):
        shifted_style_tensor = style_tensor.clone()
        shifted_style_tensor[:, channel_id] = self.style_mean[channel_id] + alpha * self.style_std[channel_id]  # [N, S]
        shifted_style_dict = style_handler.tensor2dict(shifted_style_tensor, styles_dict)
        images = self.generator.synthesis(None, encoded_styles=shifted_style_dict, noise_mode='const').clamp(-1.0, 1.0)
        images = (images + 1.0) / 2.0  # [N, C, H, W]
        image_embedds = encode_image_clip(images, self.clip_model, self.device)
        return image_embedds, images.permute(0, 2, 3, 1)

    @torch.no_grad()
    def _esitmate_relevance_scores(self):
        styles_dict = self.sample_styles_dict(self.num_samples)
        styles_handler = StyleVector(styles_dict, include_torgb=self.include_torgb_layers)
        styles_tensor = styles_handler.dict2tensor(styles_dict)  # [N, S]

        attribute_relevance_scores = []
        for ch_i in tqdm(range(styles_tensor.shape[1]), desc='Estimating channel relevance ...'):
            # predict positive embeddings
            pos_embedds, pos_img = self._predict_image_embeddings(styles_tensor, styles_handler, styles_dict,
                                                         alpha=10.0, channel_id=ch_i)
            neg_embedds, neg_img = self._predict_image_embeddings(styles_tensor, styles_handler, styles_dict,
                                                         alpha=-10.0, channel_id=ch_i)
            if False:
                print(pos_img.min(), pos_img.max())
                Image.fromarray((pos_img[1].cpu().numpy() * 255).astype(np.uint8)).save(f'pos-ch_{ch_i}.jpg')
                Image.fromarray((neg_img[1].cpu().numpy() * 255).astype(np.uint8)).save(f'neg-ch_{ch_i}.jpg')

            img_clip_delta = pos_embedds - neg_embedds
            img_clip_delta = img_clip_delta / torch.norm(img_clip_delta, dim=-1, keepdim=True)  # [N, d]

            # compute relevances
            channel_relevance = img_clip_delta @ self.delta_t
            attribute_relevance_scores.append(channel_relevance.mean().item())
        self.attribute_relevance_scores = torch.tensor(attribute_relevance_scores,
                                                       device=self.device, dtype=torch.float32)
        print('Channel relevance estimation finished successfully!')
        print(self.attribute_relevance_scores.min(), self.attribute_relevance_scores.max())

    def _compute_mean_std_style(self):
        styles_dict = self.sample_styles_dict(num_latents=100_000)
        styles = StyleVector(styles_dict, include_torgb=self.include_torgb_layers)

        style_vectors_tensor = styles.dict2tensor(styles_dict)  # [100000 x S]
        self.style_mean = torch.mean(style_vectors_tensor, dim=0)
        self.style_std = torch.std(style_vectors_tensor, dim=0)

    @torch.no_grad()
    def sample_styles_dict(self, num_latents):
        z = torch.randn(num_latents, 512, device=self.device, dtype=torch.float32)
        w_latents = self.generator.mapping(z, None)
        styles_dict = self.generator.synthesis.W2S(w_latents)
        return styles_dict

    @torch.no_grad()
    def manipulate_image(self, ws, alpha, beta):
        # find direction in the S space
        scores = torch.where(torch.abs(self.attribute_relevance_scores) >= beta,
                             self.attribute_relevance_scores,
                             torch.zeros_like(self.attribute_relevance_scores))[None]  # [1, S]
        scores /= torch.max(torch.abs(scores))

        ws = ws.to(device=self.device, dtype=torch.float32)
        styles_dict = self.generator.synthesis.W2S(ws)
        styles_handler = StyleVector(styles_dict, include_torgb=self.include_torgb_layers)
        styles_tensor = styles_handler.dict2tensor(styles_dict)  # [N, S]

        updated_styles = styles_tensor + alpha * scores * self.style_std[None]
        updated_styles_dict = styles_handler.tensor2dict(updated_styles, styles_dict)
        result = self.generator.synthesis(None, encoded_styles=updated_styles_dict, noise_mode='const')
        result = (result + 1.0) / 2.0

        return result


if __name__ == '__main__':
    STYLEGAN_PKL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl"
    NEUTRAL_TEXT = "person with straight hair"
    TARGET_TEXT = "person with curly hair"
    DEVICE = "cuda"

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    generator = LoadModel(STYLEGAN_PKL, device=DEVICE)

    styleclip = StyleClipGlobal(generator, clip_model, NEUTRAL_TEXT, TARGET_TEXT, device=DEVICE,
                                num_samples=12)
    print("StyleCLIP initialization was successful!")
