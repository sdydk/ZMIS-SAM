import torch
from torch import nn

from mmdet.registry import MODELS

from mmengine.model import BaseModule
from mmengine.dist import is_main_process

from peft import get_peft_config, get_peft_model

from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)
from .sam import ZMViTEncoder, ZHQSamMaskDecoder, ZMSamMaskDecoder
from .norm import LayerNorm2d
from torchvision import datasets, transforms
from einops import rearrange
import torch.nn.functional as F

@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class ZMISSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        shared_image_embedding = SamPositionalEmbedding(sam_config)
        if init_cfg is not None:
           from mmengine.runner.checkpoint import load_checkpoint
           load_checkpoint(
               shared_image_embedding,
               init_cfg.get('checkpoint'),
               map_location='cpu',
               revise_keys=[(r'^module\.', ''), (r'^shared_image_embedding\.', '')])
        self.shared_image_embedding = shared_image_embedding

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class ZMISSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                prompt_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^prompt_encoder\.', '')])

        self.prompt_encoder = prompt_encoder
    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class ZMISSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = ZMViTEncoder(sam_config)
        # print("===============建立vision_encoder", vision_encoder.layers[0].layer_norm1.weight)
        # 在执行Anchor之前会先进行vision_encoder的构建，以及预训练权重的添加！！！
        # load checkpoint 这一步已经将vision encoder相关的权重添加上了！！！，抛出的是maskdecoder和prompter相关权重！！！
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])
        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            # print("========================peft_config", peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class ZMISSamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)

        mask_decoder = SamMaskDecoder(sam_config)
        self.mask_decoder = mask_decoder

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)

@MODELS.register_module()
class ZMISSamHQMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        mask_decoder = ZHQSamMaskDecoder(sam_config)

        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                mask_decoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^mask_decoder\.', '')])

        self.mask_decoder = mask_decoder

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)

# 构建自己的Adapter
@MODELS.register_module()
class ZViTAdapters(BaseModule):

    def __init__(self,
                 adapter_layer,
                 embed_dim,
                 use_zooplankton_adapter_mlp=True,
                 use_zooplankton_adapter_space=True,
                 use_strip_adapter=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.adapter_layer = adapter_layer
        for idx in adapter_layer:
            self.add_module(
                f'adapter_{idx}',
                ZViTBlock(embed_dim, use_zooplankton_adapter_mlp, use_zooplankton_adapter_space, use_strip_adapter)
            )

class ZViTBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 use_zooplankton_adapter_mlp=True,
                 use_zooplankton_adapter_space=True,
                 use_strip_adapter=True,
                 ):
        super().__init__()
        # Adapter成功初始化!!!
        if use_strip_adapter:
            self.strip_adapter = StripAdapter(embed_dim, change=True)
        if use_zooplankton_adapter_space:
            self.space_adapter = Adapter(embed_dim, skip=True)
        if use_zooplankton_adapter_mlp:
            self.mlp_adapter = Adapter(embed_dim, factor=0.05)

class StripAdapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, change=False) -> None:
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        # 自适应平均此话的作用:减少特征图的空间尺寸(高度和宽度),降低计算复杂度和模型过拟合的风险
        # nn.AdaptiveAvgPool2d(output_size),其中output_size用于指定经过自适应平均池化后输出特征的尺寸,如果output_size=8,则输出特征图的尺寸为8*8
        # 考虑不用平均池化等操作，因为你把图像从[2,64,64,1280]弄成了[2,1,1,1280]，图像尺寸这么小，你再在这么小的尺度上进行带状卷积，没啥意义
        # embedding_dim=1280, hidden_dim=80
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = act_layer()
        self.fc1_1 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(1, 3), padding=(0, 1), bias=False, groups=embedding_dim)
        self.fc1_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=embedding_dim)
        self.fc2 = nn.Conv2d(embedding_dim, embedding_dim, 1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.change_channel = change

    def forward(self, x):
        if self.change_channel:
            # Z-ViT模块
            x = x.permute(0, 3, 1, 2).contiguous() 
            avg_out = self.fc2(self.fc1_2(self.act(self.fc1_1(self.avg_pool(x)))))
            max_out = self.fc2(self.fc1_2(self.act(self.fc1_1(self.max_pool(x)))))
            return self.Sigmoid(avg_out + max_out).view(x.shape[0], 1, 1, -1)
        else:
            # 提示生成器调用
            avg_out = self.fc2(self.fc1_2(self.act(self.fc1_1(self.avg_pool(x)))))
            max_out = self.fc2(self.fc1_2(self.act(self.fc1_1(self.max_pool(x)))))
            return self.Sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)

# 自己构建的Adapter已经能调用了
class Adapter(BaseModule):
    def __init__(self, embedding_dim, mlp_ratio=0.25, factor=1, act_layer=nn.GELU, skip=False,):
        super().__init__()
        self.skip = skip
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.project1 = nn.Linear(embedding_dim, hidden_dim)
        self.act = act_layer()
        self.project2 = nn.Linear(hidden_dim, embedding_dim)
        
        self.factor = factor
        self.norm = nn.LayerNorm(embedding_dim)
        self.gamma = nn.Parameter(torch.ones(embedding_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(embedding_dim))
        self.adapter_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1,)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x, hw_shapes=None):
        identity = x #[50, 14, 14, 1280]
        x1 = x.permute(0, 3, 1, 2).contiguous()
        x1 = self.relu(self.bn(self.adapter_conv(x1)))
        x1 = x1.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        # x = self.norm(x) * self.gamma + x * self.gammax #!!!!!!!!!!!!!!
        x = x1 * x
        
        # 向下映射  * self.factor
        project1 = self.project1(x)
        act = self.act(project1)
        project2 = self.project2(act)
        if self.skip:
            project2 = identity + project2 
        return project2
# 多尺度卷积模块
class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.GELU) -> None:
        # input_dim=1280, output_dim=256
        
        super().__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False,),
            LayerNorm2d(output_dim, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False,),
            LayerNorm2d(output_dim, eps=1e-6),
        )
    def forward(self, x):
        x_reshape = self.channel_reduction(x)
        return x_reshape

class ConvNeck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dim = c
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(self.dim, self.dim, 1)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.conv3 = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.conv5 = nn.Conv2d(self.dim, self.dim, 5, padding=2, groups=self.dim)
        self.conv7 = nn.Conv2d(self.dim, self.dim, 7, padding=3, groups=self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.sigmod = nn.Sigmoid()

    # def forward(self, x):
    #     x = self.act(self.bn1(self.conv1(x)))
    #     x = self.conv3(x) + self.conv5(x) + self.conv7(x)
    #     return self.act(self.bn2(self.conv1(x)))
    # 2025-09-09 19:44  加入门控机制！！
    def forward(self, x):
        identy = x
        x = self.act(self.bn1(self.conv1(x)))
        
        x3 = self.conv3(x)
        attn3 = self.sigmod(x3)
        x3 = attn3 * x + (1 - attn3) * x
        x3 = self.act(self.bn2(self.conv1(x3)))

        x5 = self.conv5(x)
        attn5 = self.sigmod(x5)
        x5 = attn5 * x + (1 - attn5) * x
        x5 = self.act(self.bn2(self.conv1(x5)))

        x7 = self.conv7(x)
        attn7 = self.sigmod(x7)
        x7 = attn7 * x + (1 - attn7) * x
        x7 = self.act(self.bn2(self.conv1(x7)))

        x = x3 + x5 + x7
        return x

class NFAM(nn.Module):
    def __init__(self, dim, n,):
        super(NFAM, self).__init__()
        self.n = n
        self.dim = dim
        self.conv_1 = nn.Conv2d(self.dim, self.dim * 2, kernel_size=3, groups=self.dim, padding=1)
        self.bn1 = nn.BatchNorm2d(self.dim * 2)
        self.gelu1 = nn.GELU()
        
        self.conv_2 = nn.Conv2d(self.dim* 2 , self.dim * 4, kernel_size=3, groups=self.dim, padding=1)
        self.bn2 = nn.BatchNorm2d(self.dim * 4)
        self.gelu2 = nn.GELU()

        self.conv_1_1 = nn.Conv2d(self.dim * 2, self.dim, kernel_size=1)
        # DSConv
        self.ds_conv = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim * 4, kernel_size=1),
            nn.BatchNorm2d(self.dim * 4),
            nn.SiLU(),  # 改为nn.GELU
            nn.Conv2d(self.dim * 4, self.dim * 4, kernel_size=3, padding=1, groups=self.dim * 4),
            nn.BatchNorm2d(self.dim * 4),
            nn.SiLU(),
            nn.Conv2d(self.dim * 4, self.dim, kernel_size=1,),
            nn.BatchNorm2d(self.dim),
        )

        self.conv1_3 = nn.Conv2d(self.dim * 4, (2 + self.n) * self.dim, kernel_size=1)
        self.conv_necks = nn.ModuleList([ConvNeck(self.dim) for _ in range(self.n)])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d((4 + n) * self.dim, self.dim, kernel_size=1),
            nn.BatchNorm2d(self.dim),
            nn.GELU(),
        )


    def forward(self, x1, x2, x3):
        # x1代表的是冻结的模块所提取的特征；x2代表的是浮游动物模块所提取的特征；x3代表的是俩拼接后的特征
        # print("===================================", self.gelu1(self.bn1(self.conv_1(x1))))
        x_1 = self.conv_1_1(self.gelu1(self.bn1(self.conv_1(x1)))) * x1   #这里原本是相加现在都改为相乘

        x_2 = self.ds_conv(self.gelu1(self.bn1(self.conv_1(x2)))) * x2

        x_branch3 = self.conv1_3(self.gelu2(self.bn2(self.conv_2(x3))))

        branches3 = torch.chunk(x_branch3, 2 + self.n, dim=1)

        processed_branches3 = []
        for i, branch in enumerate(branches3):
            if i < 2:
                processed_branches3.append(branch)
            else:
                processed_branches3.append(self.conv_necks[i - 2](branch))

        all_branches = [x_1, x_2] + processed_branches3
        x_concat = torch.cat(all_branches, dim=1)
        x_out = self.final_conv(x_concat)
        return x_out




