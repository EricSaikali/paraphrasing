# Paraphrasing

## Project Overview
This repository contains my implementation of paraphrasing models using both a simple and a complex approach.    
The main objective is to develop models that can effectively paraphrase sentences while maintaining semantic meaning, and to explore methods for evaluating paraphrasing quality.    

**The core evaluation and reasoning of my work is detailed in the `paraphrasing.ipynb`.**   

## Dataset
The project uses the Quora Question Pairs dataset, dataset of paraphrases.
## Approaches Implemented

### 1. Simple Baseline: WordNet-based Paraphraser (`models/word_net_paraphraser.py`)
- A thesaurus-based approach that performs targeted synonym substitution using WordNet
- Guided by part-of-speech tagging and lexical heuristics
- Hyperparameters optimized using grid search based on METEOR score and n-gram novelty

### 2. Complex Approach: Sequence-to-Sequence Fine-tuning (`trainers/sft_t5_base.py`)
- Fine-tuned google/flan-t5-base model via Supervised Fine-Tuning (SFT)
- Encoder-decoder architecture aligned with sequence-to-sequence tasks

## Result and Evaluation Framework (`paraphrase_evaluator.py`)

The challenge of evaluating paraphrasing quality is addressed using multiple metrics to balance competing objectives: semantic preservation, expression diversity, and fluency/grammaticality.

### Evaluation Metrics Implemented:
1. **Lexical Overlap Metrics**:
   - BLEU: Measures n-gram overlap between candidate and reference, taking into account fluency but penalizes lexical diversity.
   - ROUGE: Measures overlap focusing on recall rather than precision, evaluates textual coverage
   - METEOR: Measures similarity by aligning words based on exact matches, stems, synonyms

2. **Diversity Metrics**:
   - Edit Distance: Measures transformation cost between sentences
   - N-gram Novelty: Measures percentage of n-grams in paraphrase not in the original
   - Jaccard Similarity: Measures word overlap using set operations

3. **Semantic Evaluation**:
   - Cosine similarity via MiniLM embeddings, capturing general contextual intra and inter similarities.
   - Custom reward/paraphrase detection model fine-tuned on the Quora dataset, capturing contextual intra and inter similarities for sentences.

## Key Findings

Evaluating whether a model is "good" at paraphrasing is a complex task highlighted in results:

1. Meaning Preservation vs. Diversity: The T5 model shows significantly higher semantic similarity (+2.86%) and reward accuracy (+14.75%) but dramatically lower n-gram novelty (-28.21%). This suggests it preserves meaning better but at the cost of producing less diverse paraphrases.
2. Surface-Level vs. Semantic Metrics: While the T5 model shows higher Jaccard similarity (+19.03%) indicating greater word overlap, it performs worse on METEOR (-7.48%) and edit distance (-7.96%). This reveals the limitation of using only surface-level metrics for evaluation.
3. Model Behavior Patterns: The baseline produces more creative but less accurate paraphrases shown by a smaller BLEU score, while the T5 model creates more conservative but semantically correct rewrites. This demonstrates that different metrics favor different aspects of paraphrasing quality.
4. Detection-Based Evaluation: The significant improvement in reward accuracy (+14.75%) for the T5 model suggests that the evaluator fine-tuned for quora can capture paraphrase quality beyond traditional metrics, potentially better aligning with human judgment.
5. Practical Considerations: Despite the T5 model's high semantic scores, its lower diversity and limit usefulness in applications requiring creative rewording. This is especially true when looking at the outputs that are sometimes simple re-writes of the initial text.

These results shows that paraphrasing evaluation requires a multi-scoring system approach including meaning preservation, expression diversity, and fluency ultimately dependent on specific use case requirements.

## Repository Structure
- `models/word_net_paraphraser.py`: Implementation of the WordNet-based baseline paraphraser
- `trainers/sft_t5_base.py`: Code for fine-tuning the Flan-T5 model
- `paraphrase_evaluator.py`: Framework for evaluating paraphrasing quality
- `models/reward_model.py`: Implementation of the custom reward/evaluation model
- `trainers/reward_trainer.py`: Training code for the reward/detection model
- `dataset/quora_dataset.py`: Data handling for the Quora Question Pairs dataset
- `utils.py`: Utility file
- `storage/*`: Final trained models used

## References
1. [Paraphrase Generation: A Survey of the State of the Art](https://aclanthology.org/2021.emnlp-main.414.pdf)
2. [On the Evaluation Metrics for Paraphrase Generation](https://aclanthology.org/2022.emnlp-main.208.pdf)
3. [Paraphrase Generation as Monolingual Translation: Data and Evaluation](https://aclanthology.org/W10-4223.pdf)
4. [Evaluating n-Gram Novelty of Language Models Using Rusty-DAWG](https://aclanthology.org/2024.emnlp-main.800.pdf)
