from typing import Optional
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlora3.mypefts import PeftModelForCausalLM
from transformers.cache_utils import DynamicCache
import torch

def load_loras(
        lora_paths,
        weight_path: str = None,
        func: str = 'softmax',
        softmax_t: float = 1.0,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs
    ):
    peft_config = PeftConfig.from_pretrained(lora_paths[0], revision=None, **kwargs)
    base_model_path = peft_config.base_model_name_or_path
    base_model_revision = peft_config.revision
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, revision=base_model_revision, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        lora_paths[0], trust_remote_code=kwargs.get("trust_remote_code", False)
    )
    base_model.resize_token_embeddings(len(tokenizer))
    return PeftModelForCausalLM.from_pretrained(
        base_model,
        lora_paths,
        func=func,
        softmax_t=softmax_t,
        weight_path=weight_path,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        config=config,
        **kwargs,
    )

def train_step(
    model, 
    inputs, 
    batch, 
    first_tokens,
    scores, 
    epoch, 
    gradient_accumulation_steps,
    optimizer,
    **kwargs
):
    generation_config, model_kwargs = model._prepare_generation_config(None, **kwargs)
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
    inputs, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    device = inputs.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
    model_kwargs["use_cache"] = generation_config.use_cache
    model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
        inputs, generation_config._pad_token_tensor, generation_config._eos_token_tensor
    )
    input_ids_length = inputs.shape[-1]
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=True,
        has_default_min_length=None,
        model_input_name=model_input_name,
        inputs_tensor=inputs,
        input_ids_length=input_ids_length,
    )
    cache_name = "past_key_values"
    model_kwargs[cache_name] = DynamicCache()
    model._validate_generated_length(generation_config, input_ids_length, None)
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs,
        prefix_allowed_tokens_fn=None,
        logits_processor=[],
        device=inputs.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=[], tokenizer=None, **kwargs
    )
    inputs, model_kwargs = model._expand_inputs_for_generation(
        input_ids=inputs,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )
    return step(
        model,
        inputs,
        batch,
        first_tokens,
        scores,
        epoch,
        gradient_accumulation_steps,
        optimizer,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        **model_kwargs,
    )
    
def step(
        model,
        input_ids,
        batch,
        first_tokens,
        scores,
        epoch,
        gradient_accumulation_steps,
        optimizer,
        logits_processor,
        stopping_criteria,
        generation_config,
        **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        flag = 0
        model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
        synced_gpus = False
        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if input_ids[0, -1].detach().cpu().numpy() == 8484:
                flag = 0
            torch.cuda.empty_cache()
            if flag<first_tokens:
                model_kwargs['use_cache'] = False
                model.model.gradient_checkpointing=True
            elif flag==first_tokens:
                model.model.gradient_checkpointing=False
                model_kwargs['use_cache'] = True
                cache_name = "past_key_values"
                model_kwargs[cache_name] = DynamicCache()
                model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if flag < first_tokens:
                outputs = model(**model_inputs)
            else:
                with torch.no_grad():
                    outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if flag < first_tokens:
                # print(flag,':',-(torch.softmax(next_token_scores, dim=-1)*torch.log_softmax(next_token_scores, dim=-1)).sum().detach().cpu().numpy())
                scores += (next_token_scores,)
            flag += 1
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            del outputs
            torch.cuda.empty_cache()
            if len(scores) == batch:
                update_model(scores, epoch, gradient_accumulation_steps, optimizer)
                scores = ()
        return scores

def update_model(scores, epoch, gradient_accumulation_steps, optimizer):
    prob1 = torch.cat([torch.softmax(log, dim=-1) for log in scores])
    prob2 = torch.cat([torch.log_softmax(log, dim=-1) for log in scores])
    entropy = prob1 * prob2
    entropy = (-entropy.sum() / len(scores)) / gradient_accumulation_steps
    loss = entropy.detach().cpu().numpy()
    print("epoch: ", epoch[0], "loss: ", loss)
    
    if epoch[0] % gradient_accumulation_steps == 0:
        entropy.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        entropy.backward()
    epoch[0] += 1
    del scores
    torch.cuda.empty_cache()