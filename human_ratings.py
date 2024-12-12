import streamlit as st
import pandas as pd
from datetime import datetime
import pickle




# Load output pairs from pickle file
with open("/home/sneupane/EvaluLLM/pairs.pickle", 'rb') as file1:
    output_pairs = pickle.load(file1)
# print(output_pairs[0].keys())
# keys=output_pairs[0].keys()
# output_pairs_df = pd.DataFrame(output_pairs)
# output_pairs_df.rename(columns={"task_prompt": "Stressor Location Pair"}, inplace=True)
# internal_mapping = output_pairs_df[["input_idx","Stressor Location Pair", "model_1", "model_2"]].copy()
# internal_mapping.to_csv("internal_mapping.csv", index=False)
# # Drop unnecessary columns
# output_pairs_df = output_pairs_df.drop(columns=["input_idx", "model_1", "model_2"])

# output_pairs_df["Better_Output"] = ""  # Column for rater to choose the better output (Output_A or Output_B)
# output_pairs_df["Reason"] = ""         # Column for rater to provide reasoning

# # Save to a CSV file
# output_pairs_df.to_csv("output_pairs_for_rating.csv", index=False)


print(len(output_pairs))
df = pd.DataFrame(output_pairs)
print(len(df))
# Melt the DataFrame to create rows for each output
melted_df = df.melt(
    id_vars=["input_idx", "task_prompt","model_1", "model_2"],
    value_vars=["output1", "output2"],
    var_name="output_type",
    value_name="output_text",
)

# print(melted_df.head())

melted_df["model"] = melted_df.apply(
    lambda row: row["model_1"] if row["output_type"] == "output1" else row["model_2"], axis=1
)

# Deduplicate to ensure unique mappings
model_output_mapping = melted_df[["task_prompt", "model", "output_text"]].drop_duplicates()

# Rename columns for clarity
model_output_mapping.rename(
    columns={"task_prompt": "Stressor Location Pair", "model": "Model", "output_text": "Output"},
    inplace=True,
)
print(model_output_mapping)
# Save to a CSV file
model_output_mapping.to_csv("model_output_mapping.csv", index=False)


# Deduplicate based on input_idx and output_text
unique_outputs = melted_df.drop_duplicates(subset=["input_idx", "output_text"])

# Pivot the DataFrame to group outputs by input_idx
pivoted_df = unique_outputs.pivot_table(
    index=["input_idx", "task_prompt"],
    values="output_text",
    aggfunc=list,
).reset_index()

# Flatten the outputs into separate columns
pivoted_df = pd.concat(
    [
        pivoted_df.drop(columns=["output_text"]),
        pd.DataFrame(pivoted_df["output_text"].to_list(), index=pivoted_df.index),
    ],
    axis=1,
)
# print(pivoted_df)
# Rename columns for clarity
output_columns = [f"Output_{i+1}" for i in range(pivoted_df.shape[1] - 2)]
pivoted_df.columns = ["Input_ID", "Stressor Location Pair"] + output_columns

# Add ranking and reasoning columns
pivoted_df["Ranking (e.g., Output_1 > Output_3 > Output_2)"] = ""
pivoted_df["Reasoning"] = ""

# Save to CSV for human raters
pivoted_df.to_csv("grouped_ranking_with_reasoning.csv", index=False)
