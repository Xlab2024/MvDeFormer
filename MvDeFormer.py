import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorchs.ViT import ViT
from vit_pytorchs.DeFormer import DeFormer
import numpy as np
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class MvDeFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 768
        self.num_classes = 7
        DeFormer_range = DeFormer(
            dim=128,
            seq_len=8 * 5 ,
            depth=6,
            heads=8,
            k=20,
        )
        DeFormer_doppler = DeFormer(
            dim=128,
            seq_len=8 * 5 ,
            depth=6,
            heads=8,
            k=20,
        )
        DeFormer_angle = DeFormer(
            dim=128,
            seq_len=8 * 5 ,
            depth=6,
            heads=8,
            k=20,
        )

        self.model1 = ViT(
            # seq_pool=True,
            dim=128,
            image_size_h=128,
            image_size_w=50,
            patch_size_h=16,
            patch_size_w=10,
            channels=1,
            num_classes=768,
            transformer=DeFormer_range,
        )

        self.model2 = ViT(
            dim=128,
            image_size_h=128,
            image_size_w=50,
            patch_size_h=16,
            patch_size_w=10,
            channels=1,
            num_classes=768,
            transformer=DeFormer_doppler,
        )

        self.model3 = ViT(
            dim=128,
            image_size_h=128,
            image_size_w=50,
            patch_size_h=16,
            patch_size_w=10,
            channels=3,
            num_classes=768,
            transformer=DeFormer_angle,
        )


        self.inter_view_fusion = Linear(self.embedding_dim, 1)
        self.intra_view_fusion1 = Linear(self.embedding_dim, 1)
        self.intra_view_fusion2 = Linear(self.embedding_dim, 1)
        self.intra_view_fusion3 = Linear(self.embedding_dim, 1)
        self.fc = Linear(768, 7)


    def forward(self, data1, data2, data3,):
        output_range_vit = self.model1(data1)
        output_doppler_vit = self.model2(data2)
        output_angle_vit = self.model3(data3)

        intra_attention1 = F.softmax(self.intra_view_fusion1(output_range_vit), dim=1)
        output_range_vit = torch.matmul(intra_attention1.transpose(-1, -2), output_range_vit).squeeze(-2)

        intra_attention2 = F.softmax(self.intra_view_fusion2(output_doppler_vit), dim=1)
        output_doppler_vit = torch.matmul(intra_attention2.transpose(-1, -2), output_doppler_vit).squeeze(-2)

        intra_attention3 = F.softmax(self.intra_view_fusion3(output_angle_vit), dim=1)
        output_angle_vit = torch.matmul(intra_attention3.transpose(-1, -2), output_angle_vit).squeeze(-2)

        output_range_vit = output_range_vit.unsqueeze(-1)
        output_doppler_vit = output_doppler_vit.unsqueeze(-1)
        output_angle_vit = output_angle_vit.unsqueeze(-1)
        data = torch.cat([output_range_vit, output_doppler_vit, output_angle_vit], dim=2)
        data = data.transpose(-1, -2)

        inter_attention = F.softmax(self.inter_view_fusion(data), dim=1)
        data = torch.matmul(inter_attention.transpose(-1, -2), data).squeeze(-2)
        output = self.fc(data)
        return output

