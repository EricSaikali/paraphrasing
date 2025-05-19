import random

import nltk
from nltk.corpus import wordnet
from tqdm import tqdm


class WordNetParaphraser:
    MAX_SEEDS = 10000
    """
    baseline paraphraser that uses WordNet to replace words with synonyms.
    Based on https://aclanthology.org/W10-4223.pdf
    """

    def __init__(self, replacement_prob=0.5, max_synonyms=3, min_tok_size=4):
        self.replacement_prob = replacement_prob
        self.max_synonyms = max_synonyms
        self.min_tok_size = min_tok_size

    def get_wordnet_pos(self, treebank_tag):
        """Convert Part Of Speech tag to WordNet Part of speech tag"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def get_synonyms(self, word, pos):
        """Get synonyms for a word with the given POS"""
        synonyms = []

        if pos is None:
            return synonyms

        for synset in wordnet.synsets(word, pos=pos):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)

                    if len(synonyms) >= self.max_synonyms:
                        return synonyms

        return synonyms

    def paraphrase(self, text):
        """Generate a paraphrase by replacing words with synonyms"""
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        paraphrased_tokens = []
        for token, pos_tag in pos_tags:
            # Skip short words and punctuation
            if len(token) < self.min_tok_size or not token.isalpha():
                paraphrased_tokens.append(token)
                continue

            # Only replace with some probability
            if random.random() > self.replacement_prob:
                paraphrased_tokens.append(token)
                continue

            wordnet_pos = self.get_wordnet_pos(pos_tag)
            synonyms = self.get_synonyms(token.lower(), wordnet_pos)

            if synonyms is not None and len(synonyms) > 0:
                replacement = random.choice(synonyms)
                # Preserve capitalization
                if token[0].isupper():
                    replacement = replacement.capitalize()
                paraphrased_tokens.append(replacement)
            else:
                paraphrased_tokens.append(token)

        return ' '.join(paraphrased_tokens)

    def batch_paraphrase(self, texts, verbose=True):
        """Paraphrase a batch of texts"""
        results = []
        if verbose:
            texts_iter = tqdm(texts, desc="Generating paraphrases")
        else:
            texts_iter = texts

        for text in texts_iter:
            results.append(self.paraphrase(text))

        return results