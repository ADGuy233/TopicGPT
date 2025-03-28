import tiktoken
import openai
from typing import Callable, Optional
import numpy as np

basic_instruction =  "You are a helpful assistant. You are excellent at inferring topics from top-words extracted via topic-modelling. You make sure that everything you output is strictly based on the provided text."

class TopwordEnhancement:
    
    def __init__(
    self,
    openai_key: str,
    openai_model: str = "gpt-3.5-turbo",
    max_context_length: int = 4000,
    openai_model_temperature: float = 0.5,
    basic_model_instruction: str = basic_instruction,
    corpus_instruction: str = "",
    api_base: Optional[str] = None,
    api_type: Optional[str] = None) -> None:
        """
        Initialize the OpenAIAssistant with the specified parameters.

        Args:
            openai_key (str): Your OpenAI API key.
            openai_model (str, optional): The OpenAI model to use (default is "gpt-3.5-turbo").
            max_context_length (int, optional): The maximum length of the context for the OpenAI model (default is 4000).
            openai_model_temperature (float, optional): The softmax temperature to use for the OpenAI model (default is 0.5).
            basic_model_instruction (str, optional): The basic instruction for the model.
            corpus_instruction (str, optional): The instruction for the corpus. Useful if specific information on the corpus is available.
            api_base (str, optional): The base URL for API requests (for OpenAI compatible endpoints).
            api_type (str, optional): The type of API to use (for OpenAI compatible endpoints).

        Returns:
            None
        """

        # do some checks on the input arguments
        assert openai_key is not None, "Please provide an openai key"
        assert openai_model is not None, "Please provide an openai model"
        assert max_context_length > 0, "Please provide a positive max_context_length"
        assert openai_model_temperature > 0, "Please provide a positive openai_model_temperature"

        self.openai_key = openai_key
        self.openai_model = openai_model
        self.max_context_length = max_context_length
        self.openai_model_temperature = openai_model_temperature
        self.basic_model_instruction = basic_model_instruction
        self.corpus_instruction = f" The following information is available about the corpus used to identify the topics: {corpus_instruction}"
        self.api_base = api_base
        self.api_type = api_type
        
        # 配置OpenAI客户端
        openai.api_key = self.openai_key
        if self.api_base:
            openai.api_base = self.api_base
        if self.api_type:
            openai.api_type = self.api_type

    def __str__(self) -> str:
        repr = f"TopwordEnhancement(openai_model = {self.openai_model})"
        return repr

    def __repr__(self) -> str:
        repr = f"TopwordEnhancement(openai_model = {self.openai_model})"
        return repr
    
    def count_tokens_api_message(self, messages: list[dict[str]]) -> int:
        """
        Count the number of tokens in the API messages.

        Args:
            messages (list[dict[str]]): List of messages from the API.

        Returns:
            int: Number of tokens in the messages.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            # 处理无法自动映射到分词器的模型
            print(f"无法自动映射模型 {self.openai_model} 到分词器。使用 cl100k_base 作为默认分词器。")
            encoding = tiktoken.get_encoding("cl100k_base")
            
        n_tokens = 0
        for message in messages: 
            for key, value in message.items():
                if key == "content":
                    n_tokens += len(encoding.encode(value))
        
        return n_tokens
    
    def describe_topic_topwords_completion_object(self, 
                               topwords: list[str], 
                               n_words: int = None,
                               query_function: Callable = lambda tws: f"Please give me the common topic of those words: {tws}. Also describe the various aspects and sub-topics of the topic.") -> openai.ChatCompletion:
        """
        Describe the given topic based on its topwords using the OpenAI model.

        Args:
            topwords (list[str]): List of topwords.
            n_words (int, optional): Number of words to use for the query. If None, all words are used.
            query_function (Callable, optional): Function to query the model. The function should take a list of topwords and return a string.

        Returns:
            openai.ChatCompletion: A description of the topics by the model in the form of an OpenAI ChatCompletion object.
        """

        if n_words is None:
            n_words = len(topwords)
        
        if type(topwords) == dict:
            topwords = topwords[0]

        topwords = topwords[:n_words]
        topwords = np.array(topwords)
        
        try:
            encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            # 处理无法自动映射到分词器的模型
            print(f"无法自动映射模型 {self.openai_model} 到分词器。使用 cl100k_base 作为默认分词器。")
            encoding = tiktoken.get_encoding("cl100k_base")

        # if too many topwords are given, use only the first part of the topwords that fits into the context length
        tokens_cumsum = np.cumsum([len(encoding.encode(tw + ", ")) 
                                   for tw in topwords]) + len(encoding.encode(self.basic_model_instruction + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print("Too many topwords given. Using only the first part of the topwords that fits into the context length. Number of topwords used: ", np.argmax(tokens_cumsum > self.max_context_length))
            n_words = np.argmax(tokens_cumsum > self.max_context_length)
            topwords = topwords[:n_words]

        kwargs = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content":  self.basic_model_instruction + " " + self.corpus_instruction},
                {"role": "user", "content": query_function(topwords)},
            ],
            "temperature": self.openai_model_temperature
        }

        client = openai.OpenAI(
                api_key=self.openai_key,
                base_url=self.api_base if self.api_base else None
            )
        
        completion = client.chat.completions.create(**kwargs)

        return completion
    
    def describe_topic_topwords_str(self, 
                               topwords: list[str], 
                               n_words: int = None,
                               query_function: Callable = lambda tws: f"Please give me the common topic of those words: {tws}. Also describe the various aspects and sub-topics of the topic. Make sure the descriptions are short and concise! Do not cite more than 5 words per sub-aspect!!!") -> str:
        """
        Describe the given topic based on its topwords using the OpenAI model.

        Args:
            topwords (list[str]): List of topwords.
            n_words (int, optional): Number of words to use for the query. If None, all words are used.
            query_function (Callable, optional): Function to query the model. The function should take a list of topwords and return a string.

        Returns:
            str: A description of the topics by the model in the form of a string.
        """

        completion = self.describe_topic_topwords_completion_object(topwords, n_words, query_function)
        return completion.choices[0].message.content
    
    def generate_topic_name_str(self,
                            topwords: list[str],
                            n_words: int = None,
                            query_function: Callable = lambda tws: f"Please give me the common topic of those words: {tws}. Give me only the title of the topic and nothing else please. Make sure the title is precise and not longer than 5 words, ideally even shorter.") -> str:
        """
        Generate a topic name based on the given topwords using the OpenAI model.

        Args:
            topwords (list[str]): List of topwords.
            n_words (int, optional): Number of words to use for the query. If None, all words are used.
            query_function (Callable, optional): Function to query the model. The function should take a list of topwords and return a string.

        Returns:
            str: A topic name generated by the model in the form of a string.
        """

        return self.describe_topic_topwords_str(topwords, n_words, query_function)

    def describe_topic_documents_completion_object(self, 
                                               documents: list[str],
                                               truncate_doc_thresh=100,
                                               n_documents: int = None,
                                               query_function: Callable = lambda docs: f"Please give me the common topic of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.") -> openai.ChatCompletion:
        """
        Describe the given topic based on its documents using the OpenAI model.

        Args:
            documents (list[str]): List of documents.
            truncate_doc_thresh (int, optional): Threshold for the number of words in a document. If a document has more words than this threshold, it is pruned to this threshold.
            n_documents (int, optional): Number of documents to use for the query. If None, all documents are used.
            query_function (Callable, optional): Function to query the model. The function should take a list of documents and return a string.

        Returns:
            openai.ChatCompletion: A description of the topics by the model in the form of an openai.ChatCompletion object.
        """

        if n_documents is None:
            n_documents = len(documents)
        documents = documents[:n_documents]

        # prune documents based on number of tokens they contain 
        new_doc_lis = []
        for doc in documents:
            doc = doc.split(" ")
            if len(doc) > truncate_doc_thresh:
                doc = doc[:truncate_doc_thresh]
            new_doc_lis.append(" ".join(doc))
        documents = new_doc_lis

        try:
            encoding = tiktoken.encoding_for_model(self.openai_model)
        except KeyError:
            # 处理无法自动映射到分词器的模型
            print(f"无法自动映射模型 {self.openai_model} 到分词器。使用 cl100k_base 作为默认分词器。")
            encoding = tiktoken.get_encoding("cl100k_base")

        # if too many documents are given, use only the first part of the documents that fits into the context length
        tokens_cumsum = np.cumsum([len(encoding.encode(doc + ", ")) for doc in documents]) + len(encoding.encode(self.basic_model_instruction + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print("Too many documents given. Using only the first part of the documents that fits into the context length. Number of documents used: ", np.argmax(tokens_cumsum > self.max_context_length))
            n_documents = np.argmax(tokens_cumsum > self.max_context_length)
            documents = documents[:n_documents]
        
        kwargs = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": self.basic_model_instruction + " " + self.corpus_instruction},
                {"role": "user", "content": query_function(documents)},
            ],
            "temperature": self.openai_model_temperature
        }
        
        try:
            # 尝试使用新版API客户端
            client = openai.OpenAI(
                api_key=self.openai_key,
                base_url=self.api_base if self.api_base else None
            )
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"使用新版API客户端失败: {e}，尝试使用旧版API")
            # 使用旧版API
            completion = openai.ChatCompletion.create(**kwargs)

        return completion
    
    @staticmethod
    def sample_identity(n_docs: int) -> np.ndarray:
        """
        Generate an identity array of document indices without changing their order.

        Args:
            n_docs (int): Number of documents.

        Returns:
            np.ndarray: An array containing document indices from 0 to (n_docs - 1).
        """

        return np.arange(n_docs)
    

    @staticmethod
    def sample_uniform(n_docs: int) -> np.ndarray:
        """
        Randomly sample document indices without replacement.

        Args:
            n_docs (int): Number of documents.

        Returns:
            np.ndarray: An array containing randomly permuted document indices from 0 to (n_docs - 1).
        """

        return np.random.permutation(n_docs)
    
    @staticmethod
    def sample_poisson(n_docs: int) -> np.ndarray:
        """
        Randomly sample document indices according to a Poisson distribution, favoring documents from the beginning of the list.

        Args:
            n_docs (int): Number of documents.

        Returns:
            np.ndarray: An array containing randomly permuted document indices, with more documents drawn from the beginning of the list.
        """

        return np.random.poisson(1, n_docs)
    
    def describe_topic_documents_sampling_completion_object(
        self,
        documents: list[str],
        truncate_doc_thresh=100,
        n_documents: int = None,
        query_function: Callable = lambda docs: f"Please give me the common topic of the sample of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.",
        sampling_strategy: str = None,) -> openai.ChatCompletion:
        """
        Describe a topic based on a sample of its documents by using the openai model.

        Args:
            documents (list[str]): List of documents ordered by similarity to the topic's centroid.
            truncate_doc_thresh (int, optional): Threshold for the number of words in a document. If a document exceeds this threshold, it is truncated. Defaults to 100.
            n_documents (int, optional): Number of documents to use for the query. If None, all documents are used. Defaults to None.
            query_function (Callable, optional): Function to query the model. Defaults to a lambda function generating a query based on the provided documents.
            sampling_strategy (Union[Callable, str], optional): Strategy to sample the documents. If None, the first provided documents are used.
                If it's a string, it's interpreted as a method of the class (e.g., "sample_uniform" is interpreted as self.sample_uniform). It can also be a custom sampling function. Defaults to None.

        Returns:
            openai.ChatCompletion: A description of the topic by the model in the form of an openai.ChatCompletion object.
        """

        if type(sampling_strategy) == str:
            if sampling_strategy == "topk":
                sampling_strategy = self.sample_identity
            if sampling_strategy=="identity":
                sampling_strategy = self.sample_identity
            elif sampling_strategy=="uniform":
                sampling_strategy = self.sample_uniform
            elif sampling_strategy=="poisson":
                sampling_strategy = self.sample_poisson
        
        new_documents = [documents[i] for i in sampling_strategy(n_documents)]

        result = self.describe_topic_documents_completion_object(new_documents, truncate_doc_thresh, n_documents, query_function)
        return result
    
    def describe_topic_document_sampling_str(
    self,
    documents: list[str],
    truncate_doc_thresh=100,
    n_documents: int = None,
    query_function: Callable = lambda docs: f"Please give me the common topic of the sample of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.",
    sampling_strategy: str = None,) -> str:
        """
        Describe a topic based on a sample of its documents by using the openai model.

        Args:
            documents (list[str]): List of documents ordered by similarity to the topic's centroid.
            truncate_doc_thresh (int, optional): Threshold for the number of words in a document. If a document exceeds this threshold, it is truncated. Defaults to 100.
            n_documents (int, optional): Number of documents to use for the query. If None, all documents are used. Defaults to None.
            query_function (Callable, optional): Function to query the model. Defaults to a lambda function generating a query based on the provided documents.
            sampling_strategy (Union[Callable, str], optional): Strategy to sample the documents. If None, the first provided documents are used.
                If it's a string, it's interpreted as a method of the class (e.g., "sample_uniform" is interpreted as self.sample_uniform). It can also be a custom sampling function. Defaults to None.

        Returns:
            str: A description of the topic by the model in the form of a string.
        """

        completion = self.describe_topic_documents_sampling_completion_object(documents, truncate_doc_thresh, n_documents, query_function, sampling_strategy)
        return completion.choices[0].message.content