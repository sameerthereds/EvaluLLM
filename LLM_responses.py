from huggingface_hub import login
login("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")
import anthropic

# api_key = os.environ["ANTHROPIC_API_KEY"]

import transformers
import torch
import re
from openai import OpenAI
# Set up OpenAI API key
client_openai=  OpenAI(api_key="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",
)

claude_key="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"
client_claude = anthropic.Anthropic(api_key=claude_key)



def generate_intervention(task_prompt,model_name):
    """
    Generates a tailored stress intervention based on the provided stressor and location.
    
    Args:
        stressor (str): The source of stress.
        location (str): The location where the stressor occurs.
    
    Returns:
        str: Generated stress intervention.
    """
    prompt = (f"You are a stress intervention specialist. Suggest a practical and effective intervention "
              f"for a person experiencing stress caused by '{task_prompt[0]}' while at '{task_prompt[1]}'. "
              f"The intervention should be concise and directly address the stressor.")
    
    generated_text=""

    if model_name=="aam":
        completion = client_openai.chat.completions.create(
            model = 'gpt-4o-mini',
            messages = [
                {'role': 'user', 'content':prompt}
            ],
            temperature = 0  ,
                max_tokens=500
            )
        
        generated_text = completion.choices[0].message.content

    elif model_name=="bam":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        messages= {"role": "user", "content":prompt}

        outputs = pipeline(
            [messages],
            max_new_tokens=500,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        # print(outputs)
        # print(outputs[0]["generated_text"])
        generated_text=(outputs[0]["generated_text"][-1])
          
    elif model_name=="cam":
        message = client_claude.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.0,
        system="",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
        generated_text=message.content[0].text
    return generated_text
