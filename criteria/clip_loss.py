import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    @staticmethod
    def preprocess_image(img):
        img = F.interpolate(img, size=(224, 224), mode='bicubic')

        img = TVF.normalize(img, mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
        return img

    def forward(self, image, text):
        image = self.preprocess_image(image)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity
