import os
import getpass
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereEmbeddings

load_dotenv()

if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = getpass.getpass("Provide your HuggingFace ACCESS TOKEN: ")

def embed_hf():
    model_name = "maidalun1020/bce-embedding-base_v1"
    #model_name = "thenlper/gte-large-zh"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True}
    embedding = HuggingFaceEmbeddings(model_name=model_name,
                                     model_kwargs=model_kwargs,
                                     encode_kwargs=encode_kwargs)
    return embedding

def embed_cohere():
    model_name = "embed-multilingual-v3.0"
    embedding = CohereEmbeddings(model=model_name)
    return embedding

def rerank_hf():
    model_name = "maidalun1020/bce-reranker-base_v1"
    reranking = HuggingFaceCrossEncoder(model_name=model_name)
    return reranking


