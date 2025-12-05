import numpy as np
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
        self._mp_hands = None

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            frames, landmarks = inputs
        else:
            frames, landmarks = inputs, None

        frames = frames.contiguous()
        batch_size = frames.shape[0]
        n_frames = frames.shape[1] // self.in_planes

        if landmarks is None:
            landmarks = self._extract_landmarks_from_frames(frames, n_frames)

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

    def _init_mediapipe(self):
        if self._mp_hands is None:
            try:
                import mediapipe as mp
            except ImportError as exc:
                raise ImportError("mediapipe is required to extract landmarks on-the-fly.") from exc
            self._mp_hands = mp.solutions.hands.Hands(static_image_mode=True,
                                                      max_num_hands=1,
                                                      min_detection_confidence=0.5)

    def _extract_landmarks_from_frames(self, frames: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Run MediaPipe on raw frames to obtain landmarks when they are not precomputed."""
        self._init_mediapipe()
        b, _, h, w = frames.shape
        frames_reshaped = frames.view(b, n_frames, self.in_planes, h, w).detach().cpu()
        landmarks = torch.zeros((b, n_frames, 21, 3), dtype=torch.float32)

        for bi in range(b):
            for ti in range(n_frames):
                img = frames_reshaped[bi, ti]
                if self.in_planes >= 3:
                    img_np = img[:3].permute(1, 2, 0).numpy()
                else:
                    img_np = img[0].unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0).numpy()

                # Rescale normalized tensor to 0-255 uint8 and convert BGR->RGB for MediaPipe.
                img_np = img_np - img_np.min()
                max_val = img_np.max()
                if max_val > 0:
                    img_np = img_np / max_val
                img_np = (img_np * 255).astype(np.uint8)[..., ::-1]

                results = self._mp_hands.process(img_np)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    coords = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                    landmarks[bi, ti] = torch.tensor(coords, dtype=torch.float32)

        return landmarks.view(b, n_frames, -1).to(frames.device)


def GestureTransformerFusion(backbone: str = "resnet", in_planes: int = 3, n_classes: int = 25, **kwargs):
    if backbone not in backbone_dict:
        raise NotImplementedError("Backbone type: [{}] is not implemented.".format(backbone))
    model = _GestureTransformerFusion(backbone_dict[backbone], in_planes, n_classes, **kwargs)
    return model
