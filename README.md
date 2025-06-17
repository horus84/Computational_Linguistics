# Computational_Linguistics
# Overview

This repository contains a complete pipeline for training, evaluating, and analyzing a sequence‑to‑sequence model that transliterates Linear A sign sequences into Latin characters. It also explores data‑driven hypotheses about morphological patterns in Linear A through positional and co‑occurrence analyses, attention heatmaps, and embedding extractions.

# Key Features

    
## Data Preprocessing:

    Cleans and tokenizes raw Linear A sequences.

    Simplifies repetitive patterns and incorporates morphological tags.

## Modeling:

    Implements a PyTorch-based Seq2Seq architecture with attention.

    Trains both a baseline and a “tagged” variant that encodes prefixes/suffixes.

## Evaluation:

    Computes BLEU, exact-match accuracy, and edit distance on a held‑out test set.

    Generates quantitative reports and plots to compare model variants.

## Analysis & Visualization:

    Extracts encoder/decoder embeddings for cluster analysis.

    Produces attention heatmaps to highlight which Linear A signs influence transliterations.

    Explores co‑occurrence patterns of signs with numerals and place names to test linguistic hypotheses.

## Next Steps & Collaboration:

    Templates for integrating with domain experts.

    Guidance for iteratively refining hypotheses and expanding the tag set.

# Citation

If you use this work in your research, please cite:

    **horus84(Tanishk), “Seq2Seq Transliteration and Linguistic Analysis of the Linear A Script,” 2025.**

# Also include my IVS work too.

This repository houses a Jupyter notebook that walks through the preprocessing, visualization, and preliminary sequence‑to‑sequence experiments on the Indus Valley Script (IVS). While this is an exploratory proof‑of‑concept, it lays the groundwork for more comprehensive modeling and linguistic hypothesis testing.
Key Components

## Jupyter Notebook

        Untitled11.ipynb contains all the code cells for data loading, cleaning, tokenization, and initial modeling experiments.

        Inline plots visualize sign‑frequency distributions, co‑occurrence heatmaps, and sample attention weights.

## Dataset

        A hand‑curated JSON/CSV file of IVS sign sequences paired with proposed transliteration labels.

        Includes metadata on sequence provenance (e.g., find‑spot, inscription length).    
    
# More refernces and results could be seen here
Research: (https://www.academia.edu/129821471/Empirical_Refutation_of_Yajnadevams_Devanagari_Based_Decipherment_of_the_Indus_Script)
