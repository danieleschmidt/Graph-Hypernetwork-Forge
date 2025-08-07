"""Text processing utilities for Graph Hypernetwork Forge."""

import re
import torch
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string


# Download required NLTK data (will only download if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def preprocess_text(
    text: str, 
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_extra_spaces: bool = True,
    remove_numbers: bool = False
) -> str:
    """
    Preprocess text by cleaning and normalizing.
    
    Args:
        text: Input text string
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        remove_extra_spaces: Whether to remove extra whitespace
        remove_numbers: Whether to remove numbers
        
    Returns:
        Preprocessed text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Final cleanup of extra spaces
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def tokenize(text: str, method: str = 'word') -> List[str]:
    """
    Tokenize text into words or sentences.
    
    Args:
        text: Input text string
        method: Tokenization method ('word' or 'sentence')
        
    Returns:
        List of tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    if method == 'word':
        return word_tokenize(text)
    elif method == 'sentence':
        return sent_tokenize(text)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")


def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """
    Remove stopwords from token list.
    
    Args:
        tokens: List of word tokens
        language: Language for stopwords (default: 'english')
        
    Returns:
        List of tokens with stopwords removed
    """
    try:
        stop_words = set(stopwords.words(language))
    except OSError:
        # Fallback to common English stopwords if NLTK data unavailable
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    return [token for token in tokens if token.lower() not in stop_words]


def chunk_text(
    text: str, 
    max_length: int = 512, 
    overlap: int = 50,
    method: str = 'word'
) -> List[str]:
    """
    Split long text into overlapping chunks.
    
    Args:
        text: Input text string
        max_length: Maximum chunk length (in words or characters)
        overlap: Number of overlapping units between chunks
        method: Chunking method ('word' or 'char')
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    if method == 'word':
        tokens = text.split()
        if len(tokens) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(' '.join(chunk_tokens))
            
            if end >= len(tokens):
                break
            start = end - overlap
        
        return chunks
    
    elif method == 'char':
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append(text[start:end])
            
            if end >= len(text):
                break
            start = end - overlap
        
        return chunks
    
    else:
        raise ValueError(f"Unknown chunking method: {method}")


def calculate_similarity(text1: str, text2: str, method: str = 'tfidf') -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('tfidf', 'jaccard', 'overlap')
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    if method == 'tfidf':
        # TF-IDF cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    
    elif method == 'jaccard':
        # Jaccard similarity
        tokens1 = set(preprocess_text(text1).split())
        tokens2 = set(preprocess_text(text2).split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    elif method == 'overlap':
        # Simple word overlap
        tokens1 = set(preprocess_text(text1).split())
        tokens2 = set(preprocess_text(text2).split())
        
        intersection = len(tokens1.intersection(tokens2))
        min_len = min(len(tokens1), len(tokens2))
        
        return intersection / min_len if min_len > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def extract_keywords(
    text: str, 
    num_keywords: int = 10,
    method: str = 'tfidf',
    min_word_length: int = 2
) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text string
        num_keywords: Maximum number of keywords to extract
        method: Extraction method ('tfidf', 'frequency')
        min_word_length: Minimum word length to consider
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Preprocess text
    processed_text = preprocess_text(text, remove_punctuation=True)
    tokens = tokenize(processed_text, method='word')
    tokens = remove_stopwords(tokens)
    
    # Filter by minimum length
    tokens = [token for token in tokens if len(token) >= min_word_length]
    
    if not tokens:
        return []
    
    if method == 'frequency':
        # Simple frequency-based extraction
        token_counts = Counter(tokens)
        keywords = [word for word, count in token_counts.most_common(num_keywords)]
        return keywords
    
    elif method == 'tfidf':
        # TF-IDF based extraction
        # Create a simple corpus from the single text
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            # Fall back to frequency if we don't have multiple sentences
            return extract_keywords(text, num_keywords, method='frequency', min_word_length=min_word_length)
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_features=num_keywords * 2,  # Get more features to select from
            ngram_range=(1, 1)
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = np.argsort(mean_scores)[-num_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
        
        return keywords[:num_keywords]
    
    else:
        raise ValueError(f"Unknown keyword extraction method: {method}")


def calculate_text_statistics(text: str) -> Dict[str, Union[int, float]]:
    """
    Calculate various text statistics.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary containing text statistics
    """
    if not text:
        return {
            'num_characters': 0,
            'num_words': 0,
            'num_sentences': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'vocabulary_size': 0
        }
    
    # Basic counts
    num_characters = len(text)
    
    words = tokenize(text, method='word')
    num_words = len(words)
    
    sentences = tokenize(text, method='sentence')
    num_sentences = len(sentences)
    
    # Calculate averages
    avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0.0
    
    # Vocabulary size (unique words)
    unique_words = set(word.lower() for word in words if word.isalpha())
    vocabulary_size = len(unique_words)
    
    return {
        'num_characters': num_characters,
        'num_words': num_words,
        'num_sentences': num_sentences,
        'avg_word_length': float(avg_word_length),
        'avg_sentence_length': float(avg_sentence_length),
        'vocabulary_size': vocabulary_size
    }


def clean_text_for_embedding(text: str) -> str:
    """
    Clean text specifically for embedding models.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text suitable for embedding
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.,!?\-]', '', text)
    
    # Normalize quotes and hyphens
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'[''`]', "'", text)
    text = re.sub(r'[—–]', '-', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{2,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def truncate_text(text: str, max_length: int, method: str = 'end') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text string
        max_length: Maximum length in characters
        method: Truncation method ('end', 'middle', 'start')
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    if method == 'end':
        return text[:max_length-3] + '...'
    elif method == 'start':
        return '...' + text[-(max_length-3):]
    elif method == 'middle':
        half_len = (max_length - 3) // 2
        return text[:half_len] + '...' + text[-half_len:]
    else:
        raise ValueError(f"Unknown truncation method: {method}")


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    
    # Convert to ASCII if possible (removes accents)
    try:
        text = text.encode('ascii', 'ignore').decode('ascii')
    except:
        pass
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text