import re

import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.metrics import edit_distance
from tqdm import tqdm
import evaluate


class ParaphraseEvaluator:
    """
    Evaluates paraphrasing quality using multiple metrics:
    - Lexical: BLEU, METEOR, ROUGE
    - Semantic: Sentence embedding similarity
    - Diversity: Edit distance, n-gram novelty, Jacquard Similarity
    - Fluency: Language model perplexity (if available)
    """

    def __init__(self):
        self.bleu_scorer = evaluate.load('bleu')
        self.rouge_scorer = evaluate.load('rouge')
        self.meteor_scorer = evaluate.load('meteor')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text):
        """Tokenize text into words"""
        return self.preprocess_text(text).split()

    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score for a candidate against a reference"""
        try:
            return self.bleu_scorer.compute(predictions=[candidate], references=[reference])
        except ZeroDivisionError:
            return {'bleu': np.array(0.0)}

    def calculate_meteor(self, reference, candidate):
        """Calculate METEOR score for a candidate against a reference"""
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(candidate, str):
            candidate = [candidate]

        return self.meteor_scorer.compute(predictions=candidate, references=reference)

    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE scores for a candidate against a reference"""
        return self.rouge_scorer.compute(predictions=[candidate], references=[reference])

    def calculate_semantic_similarity(self, sentence1, sentence2):
        """Compute cosine similarity between sentence embeddings"""
        embeddings = self.embedder.encode([sentence1, sentence2])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity

    def calculate_edit_distance(self, source, paraphrase):
        """Calculate normalized word edit distance"""
        words1 = source.lower().split()
        words2 = paraphrase.lower().split()
        dist = edit_distance(words1, words2)
        max_len = max(len(words1), len(words2))
        return dist / max_len if max_len > 0 else 0

    def jaccard_similarity(self, source, paraphrase):
        """Calculate Jaccard similarity between two texts (word overlap)"""
        words1 = set(source.lower().split())
        words2 = set(paraphrase.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def calculate_ngram_novelty(self, source, paraphrase, n=3):
        """
        Calculate the fraction of n-grams in the paraphrase that are not in the source.
        Higher values indicate more novel content.
        """
        source_tokens = self.tokenize(source)
        paraphrase_tokens = self.tokenize(paraphrase)

        # Generate n-grams
        def get_ngrams(tokens, n):
            return set(' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

        source_ngrams = get_ngrams(source_tokens, n)
        paraphrase_ngrams = get_ngrams(paraphrase_tokens, n)

        if not paraphrase_ngrams:
            return 0

        # Calculate the fraction of novel n-grams
        novel_ngrams = paraphrase_ngrams - source_ngrams
        novelty = len(novel_ngrams) / len(paraphrase_ngrams)

        return novelty

    def evaluate_single(self, source, reference, paraphrase, evaluated_metric=['bleu',
                                                                               'meteor',
                                                                               'rouge',
                                                                               'edit_distance',
                                                                               'ngram_novelty',
                                                                               'jacquard_similarity',
                                                                               'semantic_similarity']):
        """Evaluate a single paraphrase against the source and reference"""
        scores = {}
        if evaluated_metric is None or len(evaluated_metric) == 0:
            return scores

        # Reference-based metrics (accuracy)
        if 'bleu' in evaluated_metric:
            bleu = self.calculate_bleu(reference, paraphrase)
            scores['bleu'] = bleu['bleu']
        if 'meteor' in evaluated_metric:
            meteor = self.calculate_meteor(reference, paraphrase)
            scores['meteor'] = meteor['meteor']
        if 'rouge' in evaluated_metric:
            rouge = self.calculate_rouge(reference, paraphrase)
            scores['rouge1'] = rouge['rouge1']
            scores['rouge2'] = rouge['rouge2']
            scores['rougeL'] = rouge['rougeL']

        # Source-based metrics (diversity)
        if 'edit_distance' in evaluated_metric:
            edit_distance = self.calculate_edit_distance(source, paraphrase)
            scores['edit_distance'] = edit_distance

        if 'ngram_novelty' in evaluated_metric:
            ngram_novelty = self.calculate_ngram_novelty(source, paraphrase, n=3)
            scores['ngram_novelty'] = ngram_novelty
        if 'jacquard_similarity' in evaluated_metric:
            jacquard_similarity = self.jaccard_similarity(source, paraphrase)
            scores['jacquard_similarity'] = jacquard_similarity

        if 'semantic_similarity' in evaluated_metric:
            scores['semantic_similarity'] = self.calculate_semantic_similarity(source, paraphrase)
        return scores

    def evaluate_batch(self,
                       sources,
                       references,
                       paraphrases,
                       evaluated_metric=['bleu',
                                         'meteor',
                                         'rouge',
                                         'edit_distance',
                                         'ngram_novelty',
                                         'jacquard_similarity',
                                         'semantic_similarity'],
                       verbose=True):
        """Evaluate a batch of paraphrases"""
        results = []
        if verbose:
            iterator = tqdm(zip(sources, references, paraphrases), total=len(sources), desc="Evaluating")
        else:
            iterator = zip(sources, references, paraphrases)

        for source, reference, paraphrase in iterator:
            eval_result = self.evaluate_single(source, reference, paraphrase, evaluated_metric)
            results.append(eval_result)

        avg_scores = {metric: np.mean([r[metric] for r in results]) for metric in results[0].keys()}

        return results, avg_scores
