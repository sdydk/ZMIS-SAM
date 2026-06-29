import torch.utils.checkpoint
from torch import nn
from transformers import SamVisionConfig, SamMaskDecoderConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoderOutput, SamVisionLayer, SamPatchEmbeddings, SamVisionNeck, SamFeedForward, SamLayerNorm, SamTwoWayTransformer
)
from typing import Optional, Tuple, Union
from .norm import LayerNorm2d


class ZMViTLayer(SamVisionLayer):

    @torch.no_grad()
    def layer_norm1_no_grad(self, x):
        return self.layer_norm1(x)

    @torch.no_grad()
    def layer_norm2_no_grad(self, x):
        return self.layer_norm2(x)

    @torch.no_grad()
    def attn_no_grad(self, hidden_states, output_attentions=False):
        return self.attn(hidden_states, output_attentions)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
            adapter: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1_no_grad(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn_no_grad(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )

        if getattr(adapter, 'space_adapter', False):
            hidden_states = adapter.space_adapter(hidden_states)

        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        if getattr(adapter, 'strip_adapter', False):
            hidden_states = hidden_states * adapter.strip_adapter(hidden_states)
        hidden_states = residual + hidden_states

        layernorm_output = self.layer_norm2_no_grad(hidden_states)

        if getattr(adapter, 'mlp_adapter', False):
            hidden_states = hidden_states + self.mlp(layernorm_output) + adapter.mlp_adapter(hidden_states)
        else:
            hidden_states = hidden_states + self.mlp(layernorm_output)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ZMViTEncoder(nn.Module):

    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = ZMViTLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    @torch.no_grad()
    def patch_embed_no_grad(self, x):
        return self.patch_embed(x)

    @torch.enable_grad()
    def patch_embed_grad(self, x):
        return self.patch_embed(x)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            adapter: Optional[torch.nn.Module] = None,
            patch_embed_grad: Optional[bool] = False,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if patch_embed_grad:
            hidden_states = self.patch_embed_grad(pixel_values)
        else:
            hidden_states = self.patch_embed_no_grad(pixel_values)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions,
                    adapter=getattr(adapter, f'adapter_{i}', None)
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs
        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



class ZMISSamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)  #需要看一下这里的上采样后的尺度是什么样的！！！
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs