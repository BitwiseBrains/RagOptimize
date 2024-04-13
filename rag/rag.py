from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

import argparse
import logging


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "hari31416/Mistral_Finance_Finetuned"
DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.1"
DEFAULT_PIPELINE_CONFIG = {
    "max_length": 1024,
    "temperature": 0.5,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
}
DEFAULT_QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
}
DEFAULT_FILE_PATH = "sample_files/wiki_options.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_simple_logger(logger_name: str, level: str = "info") -> logging.Logger:
    """Creates a simple logger with a stream handler.

    Parameters
    ----------
    logger_name : str
        The name of the logger.
    level : str
        The log level to use. Must be one of "debug", "info", "warning", "error".

    Returns
    -------
    logging.Logger
        The logger.
    """
    level_str_to_int = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger = logging.getLogger(logger_name)
    # Clear the handlers to avoid duplicate messages
    logger.handlers.clear()
    logger.setLevel(level_str_to_int[level])
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = create_simple_logger("rag", level="debug")


def create_embedding(
    file_path: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> Chroma:
    """Creates a chroma embedding of the document given in the `file_path` and the embedding model.

    Parameters
    ----------
    file_path : str
        The path to the file to create the embedding.
    embedding_model_name : str
        The name of the embedding model to use.
    chunk_size : int
        The chunk size of the embedding.
    chunk_overlap : int
        The chunk overlap of the embedding.

    Returns
    -------
    Chroma
        The chroma embedding.
    """
    logger.info(f"Creating embedding from {file_path}")
    text = TextLoader(file_path)
    document = text.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    documents = splitter.split_documents(document)
    chroma_db = Chroma.from_documents(
        documents, embedding_model, persist_directory="chroma_db"
    )
    logger.info("Embedding created")
    return chroma_db


def create_pipeline(
    llm_model_id: str = DEFAULT_LLM_MODEL,
    tokenizer_id: str = DEFAULT_TOKENIZER,
    use_quantization: bool = False,
    pipeline_config: dict = DEFAULT_PIPELINE_CONFIG,
):
    """Creates a pipeline with the LLM model given by `llm_model_id`.

    Parameters
    ----------
    llm_model_id : str
        The id of the LLM model to use.
    tokenizer_id : str
        The id of the tokenizer to use.
    use_quantization : bool
        Whether to use quantization.
    pipeline_config : dict
        The configuration of the pipeline.

    Returns
    -------
    HuggingFacePipeline
        The pipeline.
    """
    logger.info(f"Creating pipeline with LLM model {llm_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if use_quantization:
        logger.debug("Using quantization")
        quantization_config = DEFAULT_QUANTIZATION_CONFIG
    else:
        logger.debug("Not using quantization")
        quantization_config = None
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_id, quantization_config=quantization_config
    )
    if quantization_config is None:
        llm = llm.to(DEVICE)
    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        **pipeline_config,
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    logger.info("Pipeline created")
    return hf_pipeline


def _create_retrieval_qa(
    chroma_db: Chroma,
    llm_pipeline: HuggingFacePipeline,
) -> RetrievalQA:
    """Creates a retrieval QA with the given chroma embedding and LLM pipeline.

    Parameters
    ----------
    chroma_db : Chroma
        The chroma embedding.
    llm_pipeline : HuggingFacePipeline
        The LLM pipeline.

    Returns
    -------
    RetrievalQA
        The retrieval QA.
    """
    logger.info("Creating retrieval QA")
    retriever = chroma_db.as_retriever()
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
    )
    logger.info("Retrieval QA created")
    return retrieval_qa


def create_retrieval_qa(
    file_path: str = DEFAULT_FILE_PATH,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model_id: str = DEFAULT_LLM_MODEL,
    tokenizer_id: str = DEFAULT_TOKENIZER,
    use_quantization: bool = False,
    pipeline_config: dict = DEFAULT_PIPELINE_CONFIG,
) -> RetrievalQA:
    """The final function to create a retrieval QA.

    Parameters
    ----------
    file_path : str
        The path to the file to create the embedding.
    chunk_size : int
        The chunk size of the embedding.
    chunk_overlap : int
        The chunk overlap of the embedding.
    embedding_model_name : str
        The name of the embedding model to use.
    llm_model_id : str
        The id of the LLM model to use.
    tokenizer_id : str
        The id of the tokenizer to use.
    use_quantization : bool
        Whether to use quantization.
    pipeline_config : dict
        The configuration of the pipeline.

    Returns
    -------
    RetrievalQA
        The retrieval QA.
    """
    chroma = create_embedding(
        file_path,
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    llm = create_pipeline(
        llm_model_id=llm_model_id,
        tokenizer_id=tokenizer_id,
        use_quantization=use_quantization,
        pipeline_config=pipeline_config,
    )
    retrieval_qa = _create_retrieval_qa(chroma, llm)
    return retrieval_qa


def main(args: argparse.Namespace) -> None:
    """Main function.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments.
    """
    logger.debug("Starting main")
    logger.debug(f"Arguments: {args}")
    pipeline_config = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "do_sample": args.do_sample,
    }
    retrieval_qa = create_retrieval_qa(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model_name=args.embedding_model_name,
        llm_model_id=args.llm_model_id,
        tokenizer_id=args.tokenizer_id,
        use_quantization=args.use_quantization,
        pipeline_config=pipeline_config,
    )
    question = args.question
    answer = retrieval_qa.invoke(question)
    print(answer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        default=DEFAULT_FILE_PATH,
        help="Path to the file to create the embedding",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512, help="Chunk size of the embedding"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=20, help="Chunk overlap of the embedding"
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name",
    )
    parser.add_argument(
        "--llm_model_id",
        "-m",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="LLM model id",
    )
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer id",
    )
    parser.add_argument(
        "--use_quantization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use quantization",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max length of the pipeline"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature of the pipeline"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="No repeat ngram size of the pipeline",
    )
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do sample of the pipeline",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What is an option?",
        help="Question to ask",
    )
    args = parser.parse_args()
    main(args)
