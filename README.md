# RagOptimize

Fine-tuning [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) on [finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) dataset. The fine-tuned model then will be used to create a RAG to answer finance related question based on a text file. The whole system will be deployed on web.

## Fine-Tuning the Model

### Using PEFT for Fine-Tuning

Fine-tuning the complete model is not feasible and hence we have used PEFT. Our fine-tuning consists of:

- Using [QLoRA](https://arxiv.org/abs/2305.14314) for attention weights.
- Making the head trainable.

Since we are using Kaggle's GPU P100 which has only 16 GB of RAM, fine-tuning the 7B Mistral can only be done in 4 bits. Here is the LoRA config that we used:

```yaml
r: 16
lora_alpha: 4
lora_dropout: 0.05
bias: none
task_type: CAUSAL_LM
target_modules:
    - o_proj
    - v_proj
    - k_proj
    - q_proj
```

We also made the head of the model trainable and trained it along with the LoRA layers. This resulted in:

`trainable params: 144,703,488 || all params: 7,255,363,584 || trainable%: 1.9944346871755614`

The weights of the head and LoRA adapters were saved on Huggingface on regular interval, making it easier to compare various checkpoints of the model. The final weights can be found [here](https://huggingface.co/hari31416/Mistral_Finance_Finetuning/tree/11d5a558b74e9e480498ae9a220a5ae8089b0d37).

### Merging the Adapters

We need to merge the LoRA layer with the base model. Since we also trained the head of the model, the head of the base model also needs to be replaced with the fine-tuned head.

Here are the steps involved in merging the LoRA layer and the head with the base model:

- Load the base model in **full precision** (note that the model must be loaded into full precision for unloading to happen)
- Get the PEFT model using the `PeftModel.from_pretrained` by passing it the base model and the location of the LoRA layer. The latter can be done by specifying the location of the repository where the model was saved during fine-tuning.
- Call `merge_and_unload` on the PEFT model. This will merge the LoRA layer with the base model.
- Download the weights of the fine-tuned head. The weights of the head of this unloaded model than will be replaced with the downloaded weight.

## Creating RAG

## Deploying

> These sections will be filled later.