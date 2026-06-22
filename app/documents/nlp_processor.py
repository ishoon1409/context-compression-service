import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import tiktoken
import logging

logger = logging.getLogger("nlp_processor")

# Download the NLTK sentence splitter data (only runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

TOKENIZER = tiktoken.get_encoding("cl100k_base")

def calculate_token_len(text: str) -> int:
    return len(TOKENIZER.encode(text))

def extractive_summarize(text: str, target_tokens: int = 1500) -> dict:
    """
    Compresses a document using the TextRank graph algorithm.
    No LLMs are used. 100% deterministic and hallucinatory-free.
    """
    original_tokens = calculate_token_len(text)
    
    # Safety bypass for small files
    if original_tokens <= target_tokens:
        return {
            "original_token_count": original_tokens,
            "compressed_token_count": original_tokens,
            "compression_ratio": 0.0,
            "text": text
        }

    # 1. Break the document into an array of discrete sentences
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        raise ValueError("Could not extract logical sentences from text.")

    # 2. Vectorization (TF-IDF)
    # Convert sentences into a mathematical matrix based on important vocabulary
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    # 3. Graph Theory (Cosine Similarity)
    # Compare every sentence against every other sentence to find common overlap
    similarity_matrix = cosine_similarity(sentence_vectors)

    # 4. The PageRank Algorithm (TextRank)
    # Sentences that share vocabulary with other important sentences get a higher score
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # 5. Rank and Sort
    # Create a list of (Score, Sentence, Original_Index) sorted by the highest score
    ranked_sentences = sorted(
        ((scores[i], s, i) for i, s in enumerate(sentences)), 
        reverse=True
    )

    # 6. Budget Enforcement (Knapsack approach)
    selected_sentences = []
    current_tokens = 0

    for score, sentence, original_index in ranked_sentences:
        tok_len = calculate_token_len(sentence)
        
        # Stop adding sentences if it pushes us over the token budget
        if current_tokens + tok_len > target_tokens:
            continue
            
        selected_sentences.append((original_index, sentence))
        current_tokens += tok_len

    # 7. Chronological Reassembly
    # Re-sort the winning sentences by their original index so the summary reads logically
    selected_sentences.sort(key=lambda x: x[0])
    final_summary = " ".join([s for i, s in selected_sentences])
    
    compressed_tokens = calculate_token_len(final_summary)
    compression_ratio = round((1 - (compressed_tokens / original_tokens)) * 100, 2)

    return {
        "original_token_count": original_tokens,
        "compressed_token_count": compressed_tokens,
        "compression_ratio": compression_ratio,
        "text": final_summary
    }