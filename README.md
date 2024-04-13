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

The goal of this project was to create a system that can answer questions given a document. For this, we need to create a RAG system with the embedding of the text file being the knowledge base. To create a RAG, we need the following:

- An embedding of the text file.
- An LLM model.
- A retriever that can retrieve the relevant documents.

In this case, we have used [Langchain](https://www.langchain.com/) to do all the heavy lifting. The framework provides all the necessary tools to create a RAG. Here are the tools and the steps involved in creating a RAG:

- Load the text file with `langchain_community.document_loaders.TextLoader` object.
- Split the document recursively with `langchain.text_splitter. RecursiveCharacterTextSplitter` into smaller chunks of 512 tokens.
- Use [chroma](https://www.trychroma.com/) to create the embeddings of the text chunks. The model used for this is `sentence-transformers/all-MiniLM-L6-v2`. The embedding is saved locally for future use in `chroma_db` folder.
- Next, the fine-tuned model is loaded, along with the tokenizer and a `transformers.pipeline` is created for `text-generation`.
- Now, the chroma database is converted into a retriever using `as_retriever` method.
- Finally, we use the chain `langchain.chains.RetrievalQA` to create a RAG system.
- Now, one can `invoke` the RAG system with a question to get the answer.

All this is done in the [`rag.py`](https://github.com/BitwiseBrains/RagOptimize/blob/main/rag/rag.py) module. The module implements a CLI that can be used to create a RAG system and answer questions based on a text file. Have a look at the module for more details.

> Some sample text files are provided in the [`rag/sample_files`](https://github.com/BitwiseBrains/RagOptimize/blob/main/rag/sample_files) folder. The methodology to create the sample text files is included in the notebook [`rag/wiki_finance_articles.ipynb`](https://github.com/BitwiseBrains/RagOptimize/blob/main/rag/wiki_finance_articles.ipynb)

## Deploying

The RAG system can now easily be deployed on the web using [Streamlit](https://streamlit.io/) or [Gradio](https://www.gradio.app/). However, as this requires a GPU to function, since the model is quite large and inference on CPU will be painfully slow. Due to this constraint, we have not deployed the model on the web. Note that you can still run the model on your local machine using the CLI provided in the `rag.py` module.
