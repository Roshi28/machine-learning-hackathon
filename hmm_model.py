"""
Hidden Markov Model for Letter Prediction in Hangman
"""

import numpy as np
from collections import defaultdict, Counter
import pickle

class HangmanHMM:
    """
    HMM that learns letter patterns for Hangman prediction
    - Hidden states: positions in the word
    - Emissions: letters at each position
    - Uses position-based and context-based probability estimation
    """
    
    def __init__(self):
        self.letter_freq = Counter()  # Overall letter frequency
        self.position_freq = defaultdict(Counter)  # Letter frequency by position
        self.bigram_freq = defaultdict(Counter)  # Bigram patterns
        self.trigram_freq = defaultdict(Counter)  # Trigram patterns
        self.word_length_dict = defaultdict(list)  # Words grouped by length
        self.alphabet = set('abcdefghijklmnopqrstuvwxyz')
        
    def train(self, words):
        """Train HMM on corpus"""
        print("Training HMM on corpus...")
        
        for word in words:
            word = word.lower()
            length = len(word)
            self.word_length_dict[length].append(word)
            
            # Overall letter frequency
            for letter in word:
                self.letter_freq[letter] += 1
            
            # Position-based frequency
            for i, letter in enumerate(word):
                # Store both absolute position and relative position
                self.position_freq[(i, length)][letter] += 1
                
            # Bigram patterns
            for i in range(len(word) - 1):
                self.bigram_freq[word[i]][word[i+1]] += 1
            
            # Trigram patterns
            for i in range(len(word) - 2):
                self.trigram_freq[(word[i], word[i+2])][word[i+1]] += 1
        
        print(f"Trained on {len(words)} words")
        print(f"Vocabulary size: {len(self.letter_freq)} unique letters")
        print(f"Word lengths: {min(self.word_length_dict.keys())} to {max(self.word_length_dict.keys())}")
    
    def get_letter_probabilities(self, masked_word, guessed_letters):
        """
        Get probability distribution for next letter guess
        Args:
            masked_word: e.g., "_pp_e"
            guessed_letters: set of already guessed letters
        Returns:
            dict of {letter: probability}
        """
        word_length = len(masked_word)
        available_letters = self.alphabet - guessed_letters
        
        if not available_letters:
            return {}
        
        # Strategy: Combine multiple probability sources
        probs = defaultdict(float)
        
        # 1. Filter words by pattern matching
        candidate_words = self._get_candidate_words(masked_word, guessed_letters)
        
        if candidate_words:
            # Count remaining letters in candidate words
            letter_counts = Counter()
            for word in candidate_words:
                for i, letter in enumerate(word):
                    if masked_word[i] == '_' and letter not in guessed_letters:
                        letter_counts[letter] += 1
            
            # Convert to probabilities
            total = sum(letter_counts.values())
            if total > 0:
                for letter in available_letters:
                    probs[letter] += 0.7 * (letter_counts[letter] / total)
        
        # 2. Position-based probabilities
        for i, char in enumerate(masked_word):
            if char == '_':
                pos_counter = self.position_freq.get((i, word_length), Counter())
                total = sum(pos_counter.values())
                if total > 0:
                    for letter in available_letters:
                        probs[letter] += 0.2 * (pos_counter[letter] / total)
        
        # 3. Context-based (bigram/trigram)
        context_prob = self._get_context_probabilities(masked_word, available_letters)
        for letter in available_letters:
            probs[letter] += 0.1 * context_prob.get(letter, 0)
        
        # 4. Fallback to overall frequency
        total_freq = sum(self.letter_freq.values())
        for letter in available_letters:
            if probs[letter] == 0:
                probs[letter] = self.letter_freq[letter] / total_freq
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        return probs
    
    def _get_candidate_words(self, masked_word, guessed_letters):
        """Get words that match the current pattern"""
        length = len(masked_word)
        candidates = []
        
        # Get words of same length
        word_list = self.word_length_dict.get(length, [])
        
        for word in word_list:
            # Check if word matches pattern
            if self._matches_pattern(word, masked_word, guessed_letters):
                candidates.append(word)
        
        return candidates
    
    def _matches_pattern(self, word, masked_word, guessed_letters):
        """Check if word matches the masked pattern"""
        if len(word) != len(masked_word):
            return False
        
        for i, (w_char, m_char) in enumerate(zip(word, masked_word)):
            if m_char != '_':
                # Known position must match
                if w_char != m_char:
                    return False
            else:
                # Unknown position cannot be a guessed letter
                if w_char in guessed_letters:
                    return False
        
        return True
    
    def _get_context_probabilities(self, masked_word, available_letters):
        """Get probabilities based on surrounding context (bigrams/trigrams)"""
        probs = defaultdict(float)
        count = 0
        
        for i, char in enumerate(masked_word):
            if char == '_':
                # Check left context (bigram)
                if i > 0 and masked_word[i-1] != '_':
                    left_char = masked_word[i-1]
                    bigram_counter = self.bigram_freq.get(left_char, Counter())
                    total = sum(bigram_counter.values())
                    if total > 0:
                        for letter in available_letters:
                            probs[letter] += bigram_counter[letter] / total
                        count += 1
                
                # Check right context (bigram)
                if i < len(masked_word) - 1 and masked_word[i+1] != '_':
                    right_char = masked_word[i+1]
                    # Find letters that come before this right_char
                    for prev_char, counter in self.bigram_freq.items():
                        if right_char in counter:
                            total = sum(counter.values())
                            if total > 0 and prev_char in available_letters:
                                probs[prev_char] += counter[right_char] / total
                    count += 1
                
                # Check trigram context
                if i > 0 and i < len(masked_word) - 1:
                    if masked_word[i-1] != '_' and masked_word[i+1] != '_':
                        context = (masked_word[i-1], masked_word[i+1])
                        trigram_counter = self.trigram_freq.get(context, Counter())
                        total = sum(trigram_counter.values())
                        if total > 0:
                            for letter in available_letters:
                                probs[letter] += trigram_counter[letter] / total
                            count += 1
        
        # Normalize
        if count > 0:
            probs = {k: v/count for k, v in probs.items()}
        
        return probs
    
    def save(self, filename='hmm_model.pkl'):
        """Save trained model"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load(filename='hmm_model.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model


if __name__ == "__main__":
    # Test the HMM
    from hangman_solution import load_corpus
    
    corpus = load_corpus()
    
    # Train HMM
    hmm = HangmanHMM()
    hmm.train(corpus)
    
    # Test prediction
    masked_word = "_pp_e"
    guessed = set(['p', 'e'])
    probs = hmm.get_letter_probabilities(masked_word, guessed)
    
    print(f"\nTest: masked_word='{masked_word}', guessed={guessed}")
    print("Top 10 predicted letters:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for letter, prob in sorted_probs[:10]:
        print(f"  {letter}: {prob:.4f}")
    
    # Save model
    hmm.save()
