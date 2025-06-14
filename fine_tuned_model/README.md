---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:549367
- loss:SoftmaxLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: A boy in a white jersey is doing a jumping kick while another boy
    in a blue jersey is lying on the floor.
  sentences:
  - A child is outdoors.
  - The boy is practicing martial arts.
  - The small child is a girl.
- source_sentence: there are 6 horses riding on a track field with mountains in the
    background and people in the foreground watching.
  sentences:
  - There is a woman.
  - People are sleeping
  - There are several animals and people in this picture, and they are all outside.
- source_sentence: Two men and a woman sitting outside on a short wall and talking.
  sentences:
  - Several young children are playing basketball.
  - two men make a proposition to a woman
  - The man in shorts is standing up on the bus.
- source_sentence: A hairdresser in a salon looking off through the salon.
  sentences:
  - The women are playing baseball.
  - A hairdresser looks out in the salon.
  - a young person jumping from a bunk bed on a smaller bed.
- source_sentence: Two elderly people are taking a walk down a tree lined path.
  sentences:
  - People walk to work through downtown.
  - she is looking a lot of peoples
  - Two elderly people are taking a walk down a tree lined path talking about life.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Two elderly people are taking a walk down a tree lined path.',
    'Two elderly people are taking a walk down a tree lined path talking about life.',
    'she is looking a lot of peoples',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 549,367 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                              |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | int                                                                |
  | details | <ul><li>min: 6 tokens</li><li>mean: 16.44 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 10.53 tokens</li><li>max: 33 tokens</li></ul> | <ul><li>0: ~34.60%</li><li>1: ~34.20%</li><li>2: ~31.20%</li></ul> |
* Samples:
  | sentence_0                                                                             | sentence_1                                                  | label          |
  |:---------------------------------------------------------------------------------------|:------------------------------------------------------------|:---------------|
  | <code>Teenagers are partying on a boat while a blond teenager is kicking a man.</code> | <code>Teens are sitting around a late night bonfire.</code> | <code>2</code> |
  | <code>A young man dressed in a blue jacket is holding a statue over his head.</code>   | <code>A man is holding a statue over his head.</code>       | <code>0</code> |
  | <code>A boy in a striped shirt and a girl in a white shirt holding hands.</code>       | <code>A woman is kissing a man.</code>                      | <code>2</code> |
* Loss: [<code>SoftmaxLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#softmaxloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0291 | 500   | 1.0795        |
| 0.0582 | 1000  | 0.9939        |
| 0.0874 | 1500  | 0.9655        |
| 0.1165 | 2000  | 0.9629        |
| 0.1456 | 2500  | 0.9569        |
| 0.1747 | 3000  | 0.9534        |
| 0.2039 | 3500  | 0.9499        |
| 0.2330 | 4000  | 0.9514        |
| 0.2621 | 4500  | 0.9494        |
| 0.2912 | 5000  | 0.9468        |
| 0.3204 | 5500  | 0.9439        |
| 0.3495 | 6000  | 0.9423        |
| 0.3786 | 6500  | 0.9394        |
| 0.4077 | 7000  | 0.9435        |
| 0.4369 | 7500  | 0.9376        |
| 0.4660 | 8000  | 0.9401        |
| 0.4951 | 8500  | 0.9363        |
| 0.5242 | 9000  | 0.9373        |
| 0.5534 | 9500  | 0.9363        |
| 0.5825 | 10000 | 0.9361        |
| 0.6116 | 10500 | 0.9383        |
| 0.6407 | 11000 | 0.9388        |
| 0.6699 | 11500 | 0.9354        |
| 0.6990 | 12000 | 0.9358        |
| 0.7281 | 12500 | 0.9354        |
| 0.7572 | 13000 | 0.9351        |
| 0.7863 | 13500 | 0.9343        |
| 0.8155 | 14000 | 0.9327        |
| 0.8446 | 14500 | 0.9308        |
| 0.8737 | 15000 | 0.9354        |
| 0.9028 | 15500 | 0.9318        |
| 0.9320 | 16000 | 0.9312        |
| 0.9611 | 16500 | 0.9316        |
| 0.9902 | 17000 | 0.9287        |
| 1.0193 | 17500 | 0.9278        |
| 1.0485 | 18000 | 0.9234        |
| 1.0776 | 18500 | 0.9271        |
| 1.1067 | 19000 | 0.9255        |
| 1.1358 | 19500 | 0.9238        |
| 1.1650 | 20000 | 0.9291        |
| 1.1941 | 20500 | 0.9213        |
| 1.2232 | 21000 | 0.9205        |
| 1.2523 | 21500 | 0.9199        |
| 1.2815 | 22000 | 0.9201        |
| 1.3106 | 22500 | 0.9258        |
| 1.3397 | 23000 | 0.925         |
| 1.3688 | 23500 | 0.924         |
| 1.3979 | 24000 | 0.9239        |
| 1.4271 | 24500 | 0.9224        |
| 1.4562 | 25000 | 0.9207        |
| 1.4853 | 25500 | 0.9235        |
| 1.5144 | 26000 | 0.9222        |
| 1.5436 | 26500 | 0.9283        |
| 1.5727 | 27000 | 0.9221        |
| 1.6018 | 27500 | 0.9212        |
| 1.6309 | 28000 | 0.9182        |
| 1.6601 | 28500 | 0.9218        |
| 1.6892 | 29000 | 0.9221        |
| 1.7183 | 29500 | 0.9197        |
| 1.7474 | 30000 | 0.9174        |
| 1.7766 | 30500 | 0.921         |
| 1.8057 | 31000 | 0.9217        |
| 1.8348 | 31500 | 0.9205        |
| 1.8639 | 32000 | 0.9222        |
| 1.8931 | 32500 | 0.9199        |
| 1.9222 | 33000 | 0.9208        |
| 1.9513 | 33500 | 0.921         |
| 1.9804 | 34000 | 0.9193        |
| 2.0096 | 34500 | 0.916         |
| 2.0387 | 35000 | 0.9123        |
| 2.0678 | 35500 | 0.9116        |
| 2.0969 | 36000 | 0.9153        |
| 2.1260 | 36500 | 0.9147        |
| 2.1552 | 37000 | 0.9114        |
| 2.1843 | 37500 | 0.9181        |
| 2.2134 | 38000 | 0.915         |
| 2.2425 | 38500 | 0.9151        |
| 2.2717 | 39000 | 0.9161        |
| 2.3008 | 39500 | 0.9118        |
| 2.3299 | 40000 | 0.9162        |
| 2.3590 | 40500 | 0.9138        |
| 2.3882 | 41000 | 0.9118        |
| 2.4173 | 41500 | 0.9128        |
| 2.4464 | 42000 | 0.9162        |
| 2.4755 | 42500 | 0.9131        |
| 2.5047 | 43000 | 0.9144        |
| 2.5338 | 43500 | 0.9152        |
| 2.5629 | 44000 | 0.9111        |
| 2.5920 | 44500 | 0.9115        |
| 2.6212 | 45000 | 0.9119        |
| 2.6503 | 45500 | 0.9131        |
| 2.6794 | 46000 | 0.9119        |
| 2.7085 | 46500 | 0.912         |
| 2.7377 | 47000 | 0.9131        |
| 2.7668 | 47500 | 0.9146        |
| 2.7959 | 48000 | 0.9149        |
| 2.8250 | 48500 | 0.9133        |
| 2.8541 | 49000 | 0.9122        |
| 2.8833 | 49500 | 0.9118        |
| 2.9124 | 50000 | 0.9094        |
| 2.9415 | 50500 | 0.9108        |
| 2.9706 | 51000 | 0.91          |
| 2.9998 | 51500 | 0.9109        |
| 3.0289 | 52000 | 0.9075        |
| 3.0580 | 52500 | 0.9095        |
| 3.0871 | 53000 | 0.9083        |
| 3.1163 | 53500 | 0.9094        |
| 3.1454 | 54000 | 0.9076        |
| 3.1745 | 54500 | 0.9066        |
| 3.2036 | 55000 | 0.9048        |
| 3.2328 | 55500 | 0.9051        |
| 3.2619 | 56000 | 0.9078        |
| 3.2910 | 56500 | 0.9065        |
| 3.3201 | 57000 | 0.9075        |
| 3.3493 | 57500 | 0.9057        |
| 3.3784 | 58000 | 0.9076        |
| 3.4075 | 58500 | 0.9059        |
| 3.4366 | 59000 | 0.9076        |
| 3.4658 | 59500 | 0.9084        |
| 3.4949 | 60000 | 0.9067        |
| 3.5240 | 60500 | 0.9062        |
| 3.5531 | 61000 | 0.9082        |
| 3.5822 | 61500 | 0.9106        |
| 3.6114 | 62000 | 0.9059        |
| 3.6405 | 62500 | 0.9043        |
| 3.6696 | 63000 | 0.9079        |
| 3.6987 | 63500 | 0.9079        |
| 3.7279 | 64000 | 0.9047        |
| 3.7570 | 64500 | 0.905         |
| 3.7861 | 65000 | 0.9084        |
| 3.8152 | 65500 | 0.9062        |
| 3.8444 | 66000 | 0.9076        |
| 3.8735 | 66500 | 0.9077        |
| 3.9026 | 67000 | 0.9039        |
| 3.9317 | 67500 | 0.9071        |
| 3.9609 | 68000 | 0.9079        |
| 3.9900 | 68500 | 0.908         |

</details>

### Framework Versions
- Python: 3.12.7
- Sentence Transformers: 4.1.0
- Transformers: 4.52.4
- PyTorch: 2.7.1
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers and SoftmaxLoss
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->