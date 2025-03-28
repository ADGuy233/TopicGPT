import tiktoken
from tqdm import tqdm
import numpy as np
from openai import OpenAI, APIError

class GetEmbeddingsOpenAI:
    """
    This class allows to compute embeddings of text using the OpenAI API.
    """

    def __init__(self, api_key: str, embedding_model: str = "text-embedding-ada-002", 
                 tokenizer: str = 'cl100k_base', max_tokens: int = 8191, 
                 api_base: str = None, organization: str = None, num_workers: int = 10,
                 ) -> None:
        """
        Constructor of the class.

        Args:
            api_key (str): API key to use the OpenAI API.
            embedding_model (str, optional): Name of the embedding model to use.
            tokenizer (str, optional): Name of the tokenizer to use.
            max_tokens (int, optional): Maximum number of tokens to use.
            api_base (str, optional): Base URL of the API. Use this to connect to OpenAI-compatible APIs.
            organization (str, optional): Organization ID for the API.
            num_workers (int, optional): Maximum number of worker threads to use for parallel processing.

        Note:
            By default, the embedding model "text-embedding-ada-002" is used with the corresponding tokenizer "cl100k_base" and a maximum number of tokens of 8191.
        """

        self.api_key = api_key
        
        # 创建 OpenAI 客户端
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        if organization:
            client_kwargs["organization"] = organization
            
        self.client = OpenAI(**client_kwargs)
        self.embedding_model = embedding_model
        self.tokenizer_str = tokenizer
        self.max_tokens = max_tokens
        self.worker = num_workers

    @staticmethod
    def num_tokens_from_string(string: str, encoding) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): Text string to compute the number of tokens.
            encoding: A function to encode the string into tokens.

        Returns:
            int: Number of tokens in the text string.
        """
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def compute_number_of_tokens(self, corpus: list[str]) -> int:
        """
        Computes the total number of tokens needed to embed the corpus.

        Args:
            corpus (list[str]): List of strings to embed, where each element in the list is a document.

        Returns:
            int: Total number of tokens needed to embed the corpus.
        """


        if self.tokenizer_str is None:
             tokenizer = tiktoken.encoding_for_model(self.embedding_model)

        else: 
             tokenizer = tiktoken.get_encoding(self.tokenizer_str)

        num_tokens = 0
        for document in tqdm(corpus):
            num_tokens += self.num_tokens_from_string(document, tokenizer)
        
        return num_tokens
        
    def split_doc(self, text):
        """
        Splits a single document that is longer than the maximum number of tokens into a list of smaller documents.

        Args:
            self: The instance of the class.
            text (str): The string to be split.

        Returns:
            List[str]: A list of strings to embed, where each element in the list is a list of chunks comprising the document.
        """

        split_text = []
        split_text.append(text[:self.max_tokens])
        for i in range(1, len(text) // self.max_tokens):
            split_text.append(text[i * self.max_tokens:(i + 1) * self.max_tokens])
        split_text.append(text[(len(text) // self.max_tokens) * self.max_tokens:])
        return split_text
    
    def split_long_docs(self, text: list[str]) -> list[list[str]]:
        """
        Splits all documents that are longer than the maximum number of tokens into a list of smaller documents.

        Args:
            self: The instance of the class.
            text (list[str]): List of strings to embed, where each element in the list is a document.

        Returns:
            List[list[str]]: A list of lists of strings to embed, where each element in the outer list is a list of chunks comprising the document.
        """

        if self.tokenizer_str is None:
            tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        else:
            tokenizer = tiktoken.get_encoding(self.tokenizer_str)
    
         
        split_text = []
        for document in tqdm(text):
            if self.num_tokens_from_string(document, tokenizer) > self.max_tokens:
                split_text.append(self.split_doc(document))
            else:
                split_text.append([document])
        return split_text   
    
    def make_api_call(self, text: str):
        """
        Makes an API call to the OpenAI API to embed a text string.

        Args:
            self: The instance of the class.
            text (str): The string to embed.

        Returns:
            API response: The response from the API.
        """
        response = self.client.embeddings.create(
            input=[text], 
            model=self.embedding_model
        )
        return response

    def get_embeddings_doc_split(self, corpus: list[list[str]], n_tries=3, num_workers=10) -> list[dict]:
        """
        Computes the embeddings of a corpus for split documents in parallel.

        Args:
            self: The instance of the class.
            corpus (list[list[str]]): List of strings to embed, where each element is a document represented by a list of its chunks.
            n_tries (int, optional): Number of tries to make an API call (default is 3).
            num_workers (int, optional): Maximum number of worker threads (default is 10).

        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains the embedding of the document, the text of the document, and a list of errors that occurred during the embedding process.
        """
        import concurrent.futures
        
        def process_chunk(chunk_info):
            doc_idx, chunk_idx, chunk = chunk_info
            for j in range(n_tries + 1):
                try: 
                    api_res = self.make_api_call(chunk)
                    return (doc_idx, chunk_idx, {"api_res": api_res, "error": None})
                except Exception as e:
                    print(f"Error {e} occurred for chunk {chunk_idx} of document {doc_idx}")
                    print(chunk)
                    print("Trying again.")
                    if j == n_tries: 
                        print("Maximum number of tries reached. Skipping chunk.")
                        return (doc_idx, chunk_idx, {"api_res": None, "error": e})
        
        # Prepare flat list of chunks with their indices
        all_chunks = []
        for i, chunk_list in enumerate(corpus):
            for j, chunk in enumerate(chunk_list):
                all_chunks.append((i, j, chunk))
        
        # Process chunks in parallel
        chunk_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk_info): chunk_info for chunk_info in all_chunks}
            for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(all_chunks)):
                doc_idx, chunk_idx, result = future.result()
                if doc_idx not in chunk_results:
                    chunk_results[doc_idx] = {}
                chunk_results[doc_idx][chunk_idx] = result
        
        # Aggregate results by document
        api_res_list = []
        for i in range(len(corpus)):
            if i not in chunk_results:
                api_res_list.append({"embedding": None, "text": "", "errors": ["Document processing failed"]})
                continue
                
            chunk_lis = corpus[i]
            text = " ".join(chunk_lis)
            
            # Extract and aggregate embeddings
            emb_lis = []
            errors = []
            for j in range(len(chunk_lis)):
                result = chunk_results[i].get(j, {"api_res": None, "error": "Missing chunk result"})
                if result["api_res"] is not None:
                    emb_lis.append(np.array(result["api_res"].data[0].embedding))
                errors.append(result["error"])
                
            embedding = np.mean(emb_lis, axis=0) if emb_lis else None
            api_res_list.append({
                "embedding": embedding, 
                "text": text, 
                "errors": errors
            })
            
        return api_res_list
    
    def convert_api_res_list(self, api_res_list: list[dict]) -> dict:
        """
        Converts the api_res list into a dictionary containing the embeddings as a matrix and the corpus as a list of strings.

        Args:
            self: The instance of the class.
            api_res_list (list[dict]): List of dictionaries, where each dictionary contains the embedding of the document, the text of the document, and a list of errors that occurred during the embedding process.

        Returns:
            dict: A dictionary containing the embeddings as a matrix and the corpus as a list of strings.
        """


        embeddings = np.array([api_res["embedding"] for api_res in api_res_list])
        corpus = [api_res["text"] for api_res in api_res_list]
        errors = [api_res["errors"] for api_res in api_res_list]
        return {"embeddings": embeddings, "corpus": corpus, "errors": errors}

    
    def get_embeddings(self, corpus: list[str]) -> dict:
        """
        Computes the embeddings of a corpus.

        Args:
            self: The instance of the class.
            corpus (list[str]): List of strings to embed, where each element in the list is a document.

        Returns:
            dict: A dictionary containing the embeddings as a matrix and the corpus as a list of strings.
        """

        corpus_split = self.split_long_docs(corpus)
        corpus_emb = self.get_embeddings_doc_split(corpus_split, num_workers=self.worker)
        self.corpus_emb = corpus_emb
        res = self.convert_api_res_list(corpus_emb)
        return res