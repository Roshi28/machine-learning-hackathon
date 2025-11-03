"""
Enhanced Hidden Markov Model for Letter Prediction in Hangman
Implements technical guidelines:
1. Separate HMMs for different word lengths
2. Better handling of length-specific patterns
3. Improved probability estimation with smoothing
"""

import numpy as np
from collections import defaultdict, Counter
import pickle

class EnhancedHangmanHMM:
    """
    Enhanced HMM with length-specific models
    - Separate HMMs for each word length (handles different lengths better)
    - Position-based and context-based probability estimation
    - Laplace smoothing for unseen patterns
    """
    
    def __init__(self, smoothing_alpha=0.01):
        self.smoothing_alpha = smoothing_alpha
        
        # Length-specific models
        self.length_models = {}  # {length: model_data}
        
        # Global statistics (fallback)
        self.global_letter_freq = Counter()
        self.global_bigram_freq = defaultdict(Counter)
        self.global_trigram_freq = defaultdict(Counter)
        
        self.word_length_dict = defaultdict(list)
        self.alphabet = set('abcdefghijklmnopqrstuvwxyz')
        
    def train(self, words):
        """Train separate HMMs for each word length"""
        print("Training Enhanced HMM with length-specific models...")
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            word = word.lower()
            words_by_length[len(word)].append(word)
            self.word_length_dict[len(word)].append(word)
        
        # Train separate model for each length
        for length, word_list in words_by_length.items():
            self.length_models[length] = self._train_length_model(word_list, length)
            
        # Train global model as fallback
        self._train_global_model(words)
        
        print(f"Trained on {len(words)} words")
        print(f"Length-specific models: {len(self.length_models)} lengths ({min(self.length_models.keys())} to {max(self.length_models.keys())})")
        print(f"Vocabulary size: {len(self.global_letter_freq)} unique letters")
    
    def _train_length_model(self, words, length):
        """Train HMM for specific word length"""
        model = {
            'letter_freq': Counter(),
            'position_freq': defaultdict(Counter),  # [position][letter] = count
            'bigram_freq': defaultdict(Counter),
            'trigram_freq': defaultdict(Counter),
            'word_count': len(words)
        }
        
        for word in words:
            # Letter frequency
            for letter in word:
                model['letter_freq'][letter] += 1
            
            # Position-based frequency (absolute positions for this length)
            for i, letter in enumerate(word):
                model['position_freq'][i][letter] += 1
            
            # Bigram patterns
            for i in range(len(word) - 1):
                model['bigram_freq'][word[i]][word[i+1]] += 1
            
            # Trigram patterns
            for i in range(len(word) - 2):
                model['trigram_freq'][(word[i], word[i+2])][word[i+1]] += 1
        
        return model
    
    def _train_global_model(self, words):
        """Train global model for fallback"""
        for word in words:
            word = word.lower()
            
            for letter in word:
                self.global_letter_freq[letter] += 1
            
            for i in range(len(word) - 1):
                self.global_bigram_freq[word[i]][word[i+1]] += 1
            
            for i in range(len(word) - 2):
                self.global_trigram_freq[(word[i], word[i+2])][word[i+1]] += 1
    
    def get_letter_probabilities(self, masked_word, guessed_letters):
        """
        Get probability distribution using length-specific model
        """
        word_length = len(masked_word)
        available_letters = self.alphabet - guessed_letters
        
        if not available_letters:
            return {}
        
        # Get length-specific model (or use global if not available)
        model = self.length_models.get(word_length, None)
        
        if model is None:
            # Fallback to global model
            return self._get_probabilities_global(masked_word, guessed_letters, available_letters)
        
        # Use length-specific model
        return self._get_probabilities_length_specific(masked_word, guessed_letters, available_letters, model)
    
    def _get_probabilities_length_specific(self, masked_word, guessed_letters, available_letters, model):
        """Get probabilities using length-specific model"""
        probs = defaultdict(float)
        
        # Strategy 1: Pattern matching (70% weight)
        candidate_words = self._get_candidate_words(masked_word, guessed_letters)
        if candidate_words:
            letter_counts = Counter()
            for word in candidate_words:
                for i, letter in enumerate(word):
                    if masked_word[i] == '_' and letter not in guessed_letters:
                        letter_counts[letter] += 1
            
            total = sum(letter_counts.values())
            if total > 0:
                for letter in available_letters:
                    # Add Laplace smoothing
                    probs[letter] += 0.7 * (letter_counts[letter] + self.smoothing_alpha) / (total + self.smoothing_alpha * len(available_letters))
        
        # Strategy 2: Position-based probabilities (20% weight)
        for i, char in enumerate(masked_word):
            if char == '_':
                pos_counter = model['position_freq'].get(i, Counter())
                total = sum(pos_counter.values())
                if total > 0:
                    for letter in available_letters:
                        # Add Laplace smoothing
                        probs[letter] += 0.2 * (pos_counter[letter] + self.smoothing_alpha) / (total + self.smoothing_alpha * len(available_letters))
        
        # Strategy 3: Context-based (bigram/trigram) (10% weight)
        context_prob = self._get_context_probabilities(masked_word, available_letters, model)
        for letter in available_letters:
            probs[letter] += 0.1 * context_prob.get(letter, 0)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        else:
            # Fallback to uniform distribution with letter frequency bias
            total_freq = sum(model['letter_freq'].values())
            for letter in available_letters:
                probs[letter] = (model['letter_freq'][letter] + self.smoothing_alpha) / (total_freq + self.smoothing_alpha * len(available_letters))
        
        return probs
    
    def _get_probabilities_global(self, masked_word, guessed_letters, available_letters):
        """Fallback to global model"""
        probs = {}
        total_freq = sum(self.global_letter_freq.values())
        
        for letter in available_letters:
            probs[letter] = (self.global_letter_freq[letter] + self.smoothing_alpha) / (total_freq + self.smoothing_alpha * len(available_letters))
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        return probs
    
    def _get_candidate_words(self, masked_word, guessed_letters):
        """Get words that match the current pattern"""
        length = len(masked_word)
        candidates = []
        
        word_list = self.word_length_dict.get(length, [])
        
        for word in word_list:
            if self._matches_pattern(word, masked_word, guessed_letters):
                candidates.append(word)
        
        return candidates
    
    def _matches_pattern(self, word, masked_word, guessed_letters):
        """Check if word matches the masked pattern"""
        if len(word) != len(masked_word):
            return False
        
        for i, (w_char, m_char) in enumerate(zip(word, masked_word)):
            if m_char != '_':
                if w_char != m_char:
                    return False
            else:
                if w_char in guessed_letters:
                    return False
        
        return True
    
    def _get_context_probabilities(self, masked_word, available_letters, model):
        """Get probabilities based on surrounding context"""
        probs = defaultdict(float)
        count = 0
        
        for i, char in enumerate(masked_word):
            if char == '_':
                # Left context (bigram)
                if i > 0 and masked_word[i-1] != '_':
                    left_char = masked_word[i-1]
                    bigram_counter = model['bigram_freq'].get(left_char, Counter())
                    total = sum(bigram_counter.values())
                    if total > 0:
                        for letter in available_letters:
                            probs[letter] += (bigram_counter[letter] + self.smoothing_alpha) / (total + self.smoothing_alpha * len(available_letters))
                        count += 1
                
                # Trigram context
                if i > 0 and i < len(masked_word) - 1:
                    if masked_word[i-1] != '_' and masked_word[i+1] != '_':
                        context = (masked_word[i-1], masked_word[i+1])
                        trigram_counter = model['trigram_freq'].get(context, Counter())
                        total = sum(trigram_counter.values())
                        if total > 0:
                            for letter in available_letters:
                                probs[letter] += (trigram_counter[letter] + self.smoothing_alpha) / (total + self.smoothing_alpha * len(available_letters))
                            count += 1
        
        # Normalize
        if count > 0:
            probs = {k: v/count for k, v in probs.items()}
        
        return probs
    
    def save(self, filename='enhanced_hmm_model.pkl'):
        """Save trained model"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Enhanced model saved to {filename}")
    
    @staticmethod
    def load(filename='enhanced_hmm_model.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Enhanced model loaded from {filename}")
        return model


if __name__ == "__main__":
    from hangman_solution import load_corpus
    
    corpus = load_corpus()
    
    # Train Enhanced HMM
    print("Training Enhanced HMM...")
    hmm = EnhancedHangmanHMM(smoothing_alpha=0.01)
    hmm.train(corpus)
    
    # Test prediction
    test_cases = [
        ("_pp_e", set(['p', 'e'])),
        ("____", set()),
        ("_a___i_e", set(['a', 'i', 'e']))
    ]
    
    for masked_word, guessed in test_cases:
        probs = hmm.get_letter_probabilities(masked_word, guessed)
        print(f"\nTest: masked_word='{masked_word}', guessed={guessed}")
        print("Top 5 predicted letters:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for letter, prob in sorted_probs:
            print(f"  {letter}: {prob:.4f}")
    
    # Save model
    hmm.save()
