# LLM-Based Evaluation Framework for Stress Management Interventions

This repository presents an automated evaluation framework for assessing the effectiveness of large language models (LLMs) in generating stress management interventions. The framework systematically compares LLM outputs using a structured evaluation prompt, providing quantitative and qualitative insights into model performance.

## Overview

Given a dataset D, where each data instance dᵢ is a tuple (stressor, location), each instance is applied to a predefined task prompt P. The prompt instance P(dᵢ) instructs each LLM mᵢ ∈ M to generate a stress intervention tailored to the specific stressor and location context. This results in a set of generated outputs Oᵢ, where |M| represents the number of models.

The evaluation process involves:
1. **Pairwise Comparison**: A custom evaluation prompt is used to compare two generated outputs **o₁, o₂ ∈ Oᵢ**.
2. **LLM-Based Judging**: An evaluator LLM selects the preferred output based on predefined evaluation criteria **C**.
3. **Win Rate Calculation**: Each instance **P(dᵢ)** undergoes multiple evaluation trials (**nC2** comparisons, where **n** is the number of outputs per instance). Win rates are computed both per instance (local) and per model (global).

## Evaluation Methodology

The evaluation prompt is designed to elicit a binary choice from the evaluator LLM, with the structure:
- Task-specific instruction (**P** populated with variables from **dᵢ**)
- Two outputs to compare (**o₁, o₂**)
- Evaluation criteria (**C**)
- A final prompt:  
  _"Based on the evaluation criteria, the best output is [1 or 2]."_

The first non-whitespace character produced by the evaluator LLM determines the preferred output. Additionally, the evaluator generates an explanation, useful for debugging and human review.

## Scoring System

The **win rate** metric is used to quantify model performance:


This approach provides an interpretable metric that helps assess model effectiveness in real-world stress management scenarios.

## Leaderboard

| Model         | Win Rate (%) |
|--------------|-------------|
| **GPT-4o**        | 64.4%       |
| **LLaMA3-8B**     | 35.0%       |
| **Claude-Sonnet** | 0.6%        |

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/yourusername/llm-evaluation-framework.git
cd llm-evaluation-framework