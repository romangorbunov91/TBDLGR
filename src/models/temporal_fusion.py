import torch
import torch.nn as nn

from models.backbones.resnet import resnet18
from models.backbones.vgg import vgg16, vgg16_bn
from models.backbones import c3d
from models.backbones.r3d import r3d_18, r2plus1d_18
from models.attention import EncoderSelfAttention

backbone_dict = {'resnet': resnet18,
                 'vgg': vgg16, 'vgg_bn': vgg16_bn,
                 'c3d': c3d,
                 'r3d': r3d_18, 'r2plus1d': r2plus1d_18}


class MediaPipeBackbone(nn.Module):
    """Lightweight projector for MediaPipe landmarks."""

    def __init__(self, landmark_dim: int, hidden_dim: int = 128, out_dim: int = 512, dropout: float = 0.1):
        super(MediaPipeBackbone, self).__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(landmark_dim),
            nn.Linear(landmark_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, T, 21, C) or (B, T, F)
        x = x.view(x.shape[0] * x.shape[1], -1)
        return self.projector(x)


class _GestureTransformerFusion(nn.Module):
    """Two-stream model: CNN backbone fused with MediaPipe landmarks before the transformer."""

    def __init__(self, backbone: nn.Module, in_planes: int, out_planes: int,
                 pretrained: bool = False, dropout_backbone: float = 0.1,
                 landmark_dim: int = 63, landmark_hidden: int = 128, dropout_landmark: float = 0.1,
                 freeze_backbone: bool = False, **kwargs):
        super(_GestureTransformerFusion, self).__init__()

        self.in_planes = in_planes
        self.backbone_img = backbone(pretrained, in_planes, dropout=dropout_backbone)

        if freeze_backbone:
            for p in self.backbone_img.parameters():
                p.requires_grad = False

        self.backbone_mpipe = MediaPipeBackbone(
            landmark_dim=landmark_dim,
            hidden_dim=landmark_hidden,
            out_dim=512,
            dropout=dropout_landmark
        )

        self.self_attention = EncoderSelfAttention(512, 64, 64, **kwargs)
        self.fusion = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, out_planes)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            frames, landmarks = inputs
        else:
            frames, landmarks = inputs, None

        frames = frames.contiguous()
        batch_size = frames.shape[0]
        n_frames = frames.shape[1] // self.in_planes

        if landmarks is None:
            raise ValueError("Landmarks tensor is required for fusion model.")

        # CNN stream
        x_img = frames.view(-1, self.in_planes, frames.shape[-2], frames.shape[-1])
        x_img = self.backbone_img(x_img)
        x_img = x_img.view(batch_size, n_frames, -1)

        # MediaPipe stream
        x_lm = landmarks.view(batch_size, n_frames, -1)
        x_lm = self.backbone_mpipe(x_lm)
        x_lm = x_lm.view(batch_size, n_frames, -1)

        # Fusion and transformer
        x = torch.cat([x_img, x_lm], dim=-1)
        x = self.fusion(x)
        x = self.self_attention(x)

        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        x = self.classifier(x)
        return x


def GestureTransformerFusion(backbone: str = "resnet", in_planes: int = 3, n_classes: int = 25, **kwargs):
    if backbone not in backbone_dict:
        raise NotImplementedError("Backbone type: [{}] is not implemented.".format(backbone))
    model = _GestureTransformerFusion(backbone_dict[backbone], in_planes, n_classes, **kwargs)
    return model
