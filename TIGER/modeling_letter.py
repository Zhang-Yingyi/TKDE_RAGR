from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
import torch
from torch import nn
import copy
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.modeling_outputs import CausalLMOutputWithPast


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class LETTER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature


    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits/self.temperature
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):
        loss = self.ranking_loss(lm_logits, labels)             
        return loss

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class TIGER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature

    
    def gen_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):        
        loss = self.gen_loss(lm_logits, labels)             
        return loss

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class TIGER_continue(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature

    
    def gen_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):        
        loss = self.gen_loss(lm_logits, labels)             
        return loss

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )




class TIGER_continue_PPO(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature

    
    def gen_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):        
        loss = self.gen_loss(lm_logits, labels)             
        return loss

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



class TIGER_continue_embd(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature

    
    def gen_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):        
        loss = self.gen_loss(lm_logits, labels)             
        return loss

    def apply_alpha_mix(self, input_ids: torch.Tensor, alpha: float=0.1) -> torch.Tensor:
        """
        对 encoder 输入做 alpha-mix:
        假设 input_ids 的前 N 个位置是若干组 (item3, review3)，
        每 6 个 token 为一组：
            [t0, t1, t2] -> item 的 3 个 token
            [t3, t4, t5] -> review 的 3 个 token

        我这里的策略是：只替换 review 的三位：
            emb_review_mix = (1-alpha) * emb_item + alpha * emb_review
            然后把 emb_review_mix 写回 t3,t4,t5 对应的位置。

        如果你想 item 也替换，可以在下面改逻辑。
        """
        # 原始 embedding: [B, L, D]
        inputs_embeds = self.shared(input_ids)
        bsz, seqlen, hid = inputs_embeds.size()

        # 只对完整的 6 的倍数长度部分做 alpha-mix，多余尾巴不动
        num_full_groups = seqlen // 6
        if num_full_groups == 0:
            # 没有成组的 token，直接返回
            return inputs_embeds

        prefix_len = num_full_groups * 6  # 要处理的前缀长度
        prefix_embeds = inputs_embeds[:, :prefix_len, :]  # [B, 6G, D]

        # reshape 成 [B, G, 6, D]
        prefix_embeds = prefix_embeds.view(bsz, num_full_groups, 6, hid)

        # [B, G, 3, D]
        item_embeds = prefix_embeds[:, :, 0:3, :]
        review_embeds = prefix_embeds[:, :, 3:6, :]

        # alpha-mix（这里只生成“新的 review”）
        mixed_review = (1.0 - alpha) * item_embeds + alpha * review_embeds  # [B, G, 3, D]

        # 把混合后的向量写回 review 的三个位置
        prefix_embeds[:, :, 3:6, :] = mixed_review

        # reshape 回 [B, 6G, D]
        prefix_embeds = prefix_embeds.view(bsz, prefix_len, hid)

        # 把前缀替换回去，后面的 token（比如别的信息或 padding）保持不变
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, :prefix_len, :] = prefix_embeds

        return inputs_embeds
    
    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,
        alpha_mix=0.1,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        # print(alpha_mix,input_ids,inputs_embeds)
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # 如果传入了 alpha_mix，就自己先算 inputs_embeds
            if alpha_mix is not None and input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.apply_alpha_mix(input_ids)
                input_ids = None  # 用 inputs_embeds 走 encoder
                # print("\alpha = ", alpha_mix)

                # print("\alpha = ", alpha_mix)

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # # Encode if needed (training, first prediction pass)
        # if encoder_outputs is None:
        #     # Convert encoder inputs in embeddings if needed
        #     encoder_outputs = self.encoder(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         inputs_embeds=inputs_embeds,
        #         head_mask=head_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )




class TIGER_continue_embd_compress(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0
        self.sim_lambda = 0.1
        # 新增：一个简单的可学习 gate，把 sim 映射到 alpha
        
        hidden_dim = config.d_model   # 比如 768 或 1024
        gate_in = hidden_dim * 2

        # 一个简单的 2-layer MLP 做 gate
        # self.alpha_gate = nn.Sequential(
        #     nn.Linear(gate_in, hidden_dim // 2),
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid(), # 输出 (0, 1) 之间的 alpha
        # )
        # self.item_gate = nn.Sequential(
        #     nn.Linear(hidden_dim , 32),
        #     nn.LayerNorm(32),
        #     nn.Linear(32, hidden_dim),
        #     nn.Sigmoid()
        # )
        
        # 2. Review Gate: 决定 Review Embedding 保留多少信息
        self.rev_gate = nn.Sequential(
            nn.Linear(hidden_dim , 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, hidden_dim),
            nn.Sigmoid()
        )
        # nn.init.constant_(self.item_gate[-2].bias, 0.0)
        # nn.init.constant_(self.rev_gate[-2].bias, 0.0)
        # nn.init.zeros_(self.alpha_gate[-1].weight)
        # nn.init.constant_(self.alpha_gate[-1].bias, -4.0)

    def set_hyper(self,temperature):
        self.temperature = temperature

    
    def gen_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):      
        if self.training:  
            # loss = self.gen_loss(lm_logits, labels) + self.sim_lambda * getattr(self, 'align_loss', 0.0)
            loss = self.gen_loss(lm_logits, labels)
        else:
            loss = self.gen_loss(lm_logits, labels)       
        return loss
    
    def token_align_loss_cos(self, item_embeds, review_embeds, item_mask, review_mask, eps=1e-8):
        """
        【魔改】：这里不再计算简单的 cosine 距离，而是计算 InfoNCE Contrastive Loss。
        这利用了 Batch 内的其他样本作为负样本，效果远好于 MSE 或 Pairwise Cosine。
        """
        # 1. Pooling: 把 S 个 token 聚合成 1 个向量，代表整体语义
        # item_embeds: [B, G, S, D] -> [B*G, D]
        # 使用 Mean Pooling
        mask_expanded = item_mask  # [B, G, S, 1]
        
        # 聚合 Item
        item_sum = (item_embeds * mask_expanded).sum(dim=2) 
        item_cnt = mask_expanded.sum(dim=2).clamp_min(eps)
        item_vec = (item_sum / item_cnt).view(-1, item_embeds.size(-1)) # [N, D] where N = B*G

        # 聚合 Review (注意用 review_mask)
        rev_mask_expanded = review_mask
        rev_sum = (review_embeds * rev_mask_expanded).sum(dim=2)
        rev_cnt = rev_mask_expanded.sum(dim=2).clamp_min(eps)
        rev_vec = (rev_sum / rev_cnt).view(-1, review_embeds.size(-1))  # [N, D]

        # 2. Normalize
        item_vec = F.normalize(item_vec, dim=1)
        rev_vec = F.normalize(rev_vec, dim=1)

        # 3. InfoNCE 计算
        # Similarity Matrix: [N, N]
        sim_matrix = torch.matmul(item_vec, rev_vec.T) / self.temperature
        
        # Labels: 对角线是正样本 (0与0配对, 1与1配对...)
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        
        # Cross Entropy
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def token_align_loss_mse(self, item_embeds, review_embeds, item_mask, review_mask, eps=1e-6):
        # item_embeds/review_embeds: [B,G,S,D]
        # item_mask/review_mask:     [B,G,S,1]
        valid = (item_mask * review_mask).to(item_embeds.dtype)   # [B,G,S,1]
        diff2 = (item_embeds - review_embeds) ** 2                # [B,G,S,D]
        return (diff2 * valid).sum() / valid.sum().clamp_min(eps)
    
    def apply_alpha_mix_and_compress(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        alpha: float=0.3,
        sid_num: int=4,
        eps: float = 1e-6,
    ):
        # 1. 获取 Embedding
        # print(input_ids)
        inputs_embeds = self.shared(input_ids) # [B, L, D]
        bsz, seqlen, hid = inputs_embeds.size()

        # 2. 计算分组 (Unit)
        # 比如 L=14, sid=4 -> num_units=3 (3个完整块), 剩余 2 个token是真尾巴
        num_units = seqlen // sid_num
        
        # 边界：连一组都凑不齐，直接返回
        if num_units == 0:
            return inputs_embeds, attention_mask

        valid_len = num_units * sid_num
        
        # --- 核心逻辑开始 ---

        # 3. 变形：变成 [Batch, Units, sid, Dim]
        # 此时就像千层饼，每一层是一个 sid_num 长度的块
        unit_embeds = inputs_embeds[:, :valid_len, :].view(bsz, num_units, sid_num, hid)

        # 4. 切片分离 (Slicing)
        # 偶数索引 (0, 2, 4...) -> Item
        item_chunks = unit_embeds[:, 0::2, :, :] 
        # 奇数索引 (1, 3, 5...) -> Review
        rev_chunks  = unit_embeds[:, 1::2, :, :]
        item_chunks_mean = item_chunks.mean(dim=[1,2],keepdim=True)  # [B, 1, 1, D]
        rev_chunks_mean = rev_chunks.mean(dim=2,keepdim=True)    # [B, num_rev, 1, D]
        # 5. 独立过 Gate (Gating)
        # item_gate 只看 item_chunks
        # enhanced_items = item_chunks + item_chunks * self.item_gate(rev_chunks_mean)
        enhanced_items = item_chunks
        
        # rev_gate 只看 rev_chunks (如果有的话)
        # if rev_chunks.size(1) > 0:
        #     enhanced_revs = rev_chunks * self.rev_gate(rev_chunks)
        # --- Review 可能不存在，需要特殊处理 ---
        if rev_chunks.size(1) > 0:
            # 正常情况：有 Review，过 rev_gate
            rev_alpha = self.rev_gate(item_chunks_mean)
            enhanced_revs = rev_chunks + item_chunks_mean * rev_alpha
        else:
            # 【关键修复】特殊情况：只有 Item，没有 Review
            # DDP 报错原因：rev_gate 参数本轮未被使用。
            # 解决办法：拿 item_chunks 强行跑一次 rev_gate，然后 * 0
            # 这样计算图里就有了 rev_gate，但结果不影响数值
            dummy_out = self.rev_gate(item_chunks_mean) # 跑一次，为了产生梯度路径
            enhanced_revs = rev_chunks # 依然是空的
            
            # 把这个 dummy 的梯度“挂”到 enhanced_items 上
            # 0 * dummy_out 结果全是 0，不会改变 item 的值，但连接了计算图
            enhanced_items = enhanced_items + (0 * dummy_out)
        # enhanced_revs = rev_chunks
        # 6. 拉链式合并 (Merging)
        # 创建空容器
        output_head = torch.zeros_like(unit_embeds)
        
        # 填回 Item (包含落单的那个 Item)
        output_head[:, 0::2, :, :] = enhanced_items
        
        # 填回 Review (如果有)
        if rev_chunks.size(1) > 0:
            output_head[:, 1::2, :, :] = enhanced_revs
            
        # 展平头部: [B, valid_len, D]
        output_head = output_head.view(bsz, valid_len, hid)

        # 7. 拼接真正的尾巴 (Tail Concatenation)
        # 把刚才剩下的那几个 token (凑不齐 sid_num 的) 拼回来
        if valid_len < seqlen:
            tail_embeds = inputs_embeds[:, valid_len:, :]
            final_embeds = torch.cat([output_head, tail_embeds], dim=1)
        else:
            final_embeds = output_head

        # Mask 全程不需要动，因为 Input 和 Output 长度完全一致
        return final_embeds, attention_mask
    
    def apply_alpha_mix_and_compress_old(
        self,
        input_ids: torch.Tensor,        # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        alpha: float=0.3,
        sid_num: int=4,
        eps: float = 1e-6,
    ):
        """
        修正版：
        1. 保持序列长度不变 (No Compress)。
        2. 使用 Review 的整体语义来增强 Item (Injection)。
        3. 输出格式: [Enhanced_Item, Original_Review, Tail...]
        """

        # 原始 embedding
        inputs_embeds = self.shared(input_ids)  # [B, L, D]
        bsz, seqlen, hid = inputs_embeds.size()
        group_len = 2 * sid_num
        num_full_groups = seqlen // group_len

        # 边界情况处理
        if num_full_groups == 0:
            self.align_loss = torch.tensor(0.0).to(inputs_embeds.device)
            return inputs_embeds, attention_mask

        prefix_len = num_full_groups * group_len
        
        # 1. 拆分 Item 和 Review
        # [B, G, 2*sid, D]
        prefix_embeds = inputs_embeds[:, :prefix_len, :].view(bsz, num_full_groups, group_len, hid)
        prefix_mask = attention_mask[:, :prefix_len].view(bsz, num_full_groups, group_len)

        # Item: [B, G, sid, D]
        item_embeds   = prefix_embeds[:, :, 0:sid_num, :]
        review_embeds = prefix_embeds[:, :, sid_num:2*sid_num, :]

        # Mask: [B, G, sid, 1]
        item_mask   = prefix_mask[:, :, 0:sid_num].unsqueeze(-1).to(item_embeds.dtype)
        review_mask = prefix_mask[:, :, sid_num:2*sid_num].unsqueeze(-1).to(review_embeds.dtype)

        # 2. 计算 Loss (InfoNCE)
        if self.training:
            self.align_loss = self.token_align_loss_cos(item_embeds, review_embeds, item_mask, review_mask)
        else:
            self.align_loss = torch.tensor(0.0).to(inputs_embeds.device)


        # 4. Gate 计算 (Item + Review_Context)
        # 这样 Gate 知道："对于这个 Category ID，结合这条好评，我该注入多少信息？"
        gate_input = torch.cat([item_embeds, review_embeds], dim=-1) # [B, G, sid, 2D]
        alpha_logits = self.alpha_gate(gate_input) # [B, G, sid, 1]
        
        # 5. 注入增强 (Residual Connection)
        # enhanced_item = Item + alpha * Review_Context
        # 这样保留了 ID 及其位置信息，同时注入了 Review 的语义
        enhanced_item_embeds = (1+alpha_logits) * item_embeds 
        # enhanced_review_embeds = review_embeds + alpha_logits * item_embeds
        enhanced_review_embeds = (1- alpha_logits) * review_embeds

        # 6. 重新拼接 [Enhanced_Item, Original_Review]
        # [B, G, 2*sid, D]
        final_prefix_embeds = torch.cat([enhanced_item_embeds, enhanced_review_embeds], dim=2)
        
        # 展平回 [B, prefix_len, D]
        final_prefix_embeds = final_prefix_embeds.view(bsz, prefix_len, hid)
        
        # 处理 Mask (Item mask 和 Review mask 都要保留)
        final_prefix_mask = prefix_mask.view(bsz, prefix_len) # 直接复用原始的 mask 即可，因为长度没变

        # 7. 处理尾部 (Tail)
        if prefix_len < seqlen:
            tail_embeds = inputs_embeds[:, prefix_len:, :]
            tail_mask = attention_mask[:, prefix_len:]
            
            final_embeds = torch.cat([final_prefix_embeds, tail_embeds], dim=1)
            final_mask = torch.cat([final_prefix_mask, tail_mask], dim=1)
        else:
            final_embeds = final_prefix_embeds
            final_mask = final_prefix_mask

        return final_embeds, final_mask
    
    
    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,
        alpha_mix=0.1,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        # print(alpha_mix,input_ids,inputs_embeds)
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # 如果传入了 alpha_mix，就自己先算 inputs_embeds
            if alpha_mix is not None and input_ids is not None and inputs_embeds is None:
                inputs_embeds, attention_mask = self.apply_alpha_mix_and_compress(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                input_ids = None  # 后面 encoder 只用 embeds
                # print("\alpha = ", alpha_mix)

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    