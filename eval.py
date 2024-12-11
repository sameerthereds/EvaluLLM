import random
import itertools
from collections import Counter
from LLM_responses import *
import pandas as pd
def generate_model_output(model_name, task_prompt):
    if model_name != "":
        return generate_intervention(task_prompt,model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    


def generate_synthetic_outputs(input_data):
    models = ["aam", "bam","cam"]
    outputs = {model: [] for model in models}
    for task_prompt in input_data:
        for model in models:
            model_outputs=generate_model_output(model, task_prompt)
            if "content" in model_outputs:
                print(model)
                model_outputs=model_outputs["content"]
            outputs[model].append(model_outputs)
    return outputs




def generate_output_pairs(outputs):
    models = list(outputs.keys())
    pairwise_comparisons = []
    for i, task_prompt in enumerate(input_data):
        pairs = itertools.combinations(models, 2)
        for model1, model2 in pairs:
            pairwise_comparisons.append({
                "input_idx": i,
                "task_prompt": task_prompt,
                "model_1": model1,
                "output1": outputs[model1][i],
                "model_2": model2,
                "output2": outputs[model2][i]
            })
    return pairwise_comparisons




def evaluate_pairs(pairwise_comparisons):
    evaluation_results = []
    for pair in pairwise_comparisons:
        # print(pair['task_prompt'])
        eval_prompt = (
            f"Given the task prompt: '{pair['task_prompt']}'\n\n"
            f"Output from {pair['model_1']}: '{pair['output1']}'\n"
            f"Output from {pair['model_2']}: '{pair['output2']}'\n\n"
            f"Which output is better in terms of addressing the {pair['task_prompt'][0]} occured at {pair['task_prompt'][1]} effectively and why? Provide the preferred output and rationale."
        )
        completion = client_openai.chat.completions.create(
          model = 'gpt-4o-mini',
          messages = [
            {'role': 'user', 'content':eval_prompt}
          ],
          temperature = 0  ,
            max_tokens=500
        )
    
        rationale = completion.choices[0].message.content.strip()
        rationale=rationale.lower()
        # print(rationale)
        preferred_model = pair["model_1"] if pair["model_1"] in rationale else pair["model_2"]
        # print(preferred_model)
        evaluation_results.append({
            "input_idx": pair["input_idx"],
            "pair": (pair["model_1"], pair["model_2"]),
            "preferred_model": preferred_model,
            "rationale": rationale
        })
    return evaluation_results



def calculate_leaderboard(evaluation_results,model_dict):
    win_count = Counter([model_dict[result["preferred_model"]] for result in evaluation_results])
    leaderboard = sorted(win_count.items(), key=lambda x: x[1], reverse=True)
    return leaderboard

import os
print("Current working directory:", os.getcwd())
stressor_df=pd.read_csv("/home/sneupane/stressLLM/EvaluLLM/moods_stressor_data.csv")
stressor_group_df=stressor_df.groupby(["mod_stressor_new","Location"],as_index=False)["user"].count().sort_values("user",ascending=False)
test_data=stressor_group_df.head(3)
tuple_list = list(zip(test_data['mod_stressor_new'], test_data['Location']))
input_data = tuple_list

outputs = generate_synthetic_outputs(input_data)




pairs = generate_output_pairs(outputs)

# pairs = generate_output_pairs(outputs)
results = evaluate_pairs(pairs)


leaderboard = calculate_leaderboard(results,model_dict={"aam":"openai","bam":"llama","cam":"claude"})

# leaderboard=[(model_dict[i[0]]for i in leaderboard)]
import pickle


with open('outputs.pickle', 'wb') as handle:
    pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pairs.pickle', 'wb') as handle:
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("\nEvaluation Results:")
# for result in results:
#     print(result)



print("\nModel Leaderboard (Win Count):")
print(leaderboard)