# data/augmentation.py

import random
import nltk
from nltk.corpus import wordnet as wn
from typing import Dict

# Download WordNet if not already installed
nltk.download('wordnet', quiet=True)


def synonym_substitution(text: str) -> str:
    """
    Substitute words in the text with their synonyms to create a similar meaning.
    Ensures the substitutions maintain the original context.
    """
    words = text.split()
    augmented_words = []

    for word in words:
        # Find synonyms for the word using WordNet
        synonyms = wn.synsets(word)

        # Only substitute if synonyms are available
        if synonyms:
            # Pick a synonym that shares the same part of speech as the original word
            synonym = synonyms[0].lemmas()[0].name()  # Use the first synonym lemma
            augmented_words.append(synonym if synonym != word else word)
        else:
            augmented_words.append(word)

    return ' '.join(augmented_words)


def antonym_substitution(text: str) -> str:
    """
    Substitute words in the text with their antonyms to create a contradictory meaning.
    Only substitutes words when antonyms are available in WordNet.
    """
    words = text.split()
    augmented_words = []

    for word in words:
        antonyms = []

        # Search for antonyms in WordNet
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())

        # Substitute with a randomly chosen antonym if any are available
        if antonyms:
            antonym = random.choice(antonyms)
            augmented_words.append(antonym)
        else:
            augmented_words.append(word)

    return ' '.join(augmented_words)


def augment_sample(example: Dict[str, str]) -> Dict[str, str]:
    """
    Apply augmentations to a given example to create adversarial data.
    - Synonym substitution for `ENTAILMENT` and `NEUTRAL` labels
    - Antonym substitution for `CONTRADICTION` labels
    """
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    # Apply augmentations based on label
    if label == "ENTAILMENT":
        # For entailment, use synonym substitution on the hypothesis
        new_hypothesis = synonym_substitution(hypothesis)
        new_label = "ENTAILMENT"
    elif label == "NEUTRAL":
        # For neutral examples, also use synonym substitution on the hypothesis
        new_hypothesis = synonym_substitution(hypothesis)
        new_label = "NEUTRAL"
    elif label == "CONTRADICTION":
        # For contradictions, apply antonym substitution to flip meaning in hypothesis
        new_hypothesis = antonym_substitution(hypothesis)
        new_label = "CONTRADICTION"
    else:
        raise ValueError(f"Unexpected label {label} found in data.")

    # Return the augmented sample as a dictionary
    return {
        "premise": premise,
        "hypothesis": new_hypothesis,
        "label": new_label
    }


def augment_dataset(dataset):
    """
    Apply the augment_sample function to each entry in the dataset,
    returning an augmented dataset.
    """
    return dataset.map(augment_sample)
