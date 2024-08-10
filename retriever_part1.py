import pickle
import urllib

from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def create_retriever() -> VectorStoreRetriever:
  """
  Creates a vector store index populated with movie reviews base on the tutorial
  https://medium.com/thedeephub/pair-programming-an-llm-app-4ca1629295d0
  Note: the code bellow uses as starting point a pickle containing the 
  processed data obtained at the end of section 2 of the previous tutorial,
  this is to avoid repeating too much code.
  """
  MOVIES_PKL_URL = "https://tutorials-public.s3.eu-west-1.amazonaws.com/" + \
    "llm_pair_programming_series/movie_reviews_processed.pkl"
  with urllib.request.urlopen(MOVIES_PKL_URL) as response:
      movie_reviews = pickle.load(response)

  EMBEDDING_MODEL_NAME = "text-embedding-3-large"

  embeder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      length_function=len,
  )

  index = VectorstoreIndexCreator(
      vectorstore_cls=DocArrayInMemorySearch,
      embedding=embeder,
      text_splitter=text_splitter,
  ).from_documents(movie_reviews)

  return index.vectorstore.as_retriever()
