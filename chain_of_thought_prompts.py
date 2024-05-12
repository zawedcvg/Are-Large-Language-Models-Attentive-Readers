import json
from datasets import load_from_disk, load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import random
from tqdm import tqdm

random.seed(42)

def chain_of_thought_prompt(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset):
        context = datapoint['context']['sentences']
        total_context = []

        for topic in context:
            total_context.append("".join(topic))
            context_thing = "".join(total_context)

        prompt_multi = f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context.\
                You will break the questions into sub questions. You will then use these sub-questions to get to the final answer. The final answer must be have 'Final Answer: ' prepended to it . Thus your output will be in the following format:\nSub-question 1: [subquestion 1]\nAnswer: [answer 1]\nsub-question 2: [subquestion 2]\nAnswer: [answer 2]\nSub-question n: [subquestion n]\nAnswer: [answer n]\nFinal Answer: [final answer]. The final answer should be limited to 5 words with just the answer and no explanation/information.\n\nHere are some past conversations:\nContext: ......\nQuestion: \'Which government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell\'?\nSub-question 1: Which woman portrayed Corliss Archer in the film Kiss and Tell?.\nAnswer: Shirley Temple.\nSub-question-2 : Which government position was held by Shirley Temple?\nAnswer: Chief of Protocol\nFinal Answer: Chief of Protocol.\n\n\
Context : .......\n\
Question: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?\n\
Sub-question 1: Which university has its main campus in Lawrence, Kansas and has branch campuses in Kansas City metropolitan area?\n\
Answer: University of Kansas\n\
Sub-question 2: What is the name of the fight song of University of Kansas?\n\
Answer: Kansas Song\n\
Final Answer: Kansas Song\n\
\n<</SYS>>\n\n\
Context: {context_thing}\n\
Question: {datapoint['question']}[/INST]"
        json_content.append({'prompt': prompt_multi})
        # print(prompt_multi)
        count += 1
    return json_content

def chain_of_thought_prompt_subqa(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset, total=len(dataset)):
        context = [i[1] for i in datapoint['context']]
        total_context = []

        for topic in context:
            # print(topic)
            # print("here")
            total_context.append("".join(topic))
            context_thing = "".join(total_context)

        prompt_multi = f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context.\
                You will break the questions into sub questions. You will then use these sub-questions to get to the final answer. The final answer must be have 'Final Answer: ' prepended to it . Thus your output will be in the following format:\nSub-question 1: [subquestion 1]\nAnswer: [answer 1]\nsub-question 2: [subquestion 2]\nAnswer: [answer 2]\nSub-question n: [subquestion n]\nAnswer: [answer n]\nFinal Answer: [final answer]. The final answer should be limited to 5 words with just the answer and no explanation/information.\n\nHere are some past conversations:\nContext: ......\nQuestion: \'Which government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell\'?\nSub-question 1: Which woman portrayed Corliss Archer in the film Kiss and Tell?.\nAnswer: Shirley Temple.\nSub-question-2 : Which government position was held by Shirley Temple?\nAnswer: Chief of Protocol\nFinal Answer: Chief of Protocol.\n\n\
Context : .......\n\
Question: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?\n\
Sub-question 1: Which university has its main campus in Lawrence, Kansas and has branch campuses in Kansas City metropolitan area?\n\
Answer: University of Kansas\n\
Sub-question 2: What is the name of the fight song of University of Kansas?\n\
Answer: Kansas Song\n\
Final Answer: Kansas Song\n\
\n<</SYS>>\n\n\
Context: {context_thing}\n\
Question: {datapoint['question']}[/INST]"
        json_content.append({'prompt': prompt_multi})
        # print(prompt_multi)
        count += 1
    return json_content
def chain_of_thought_prompt_control(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset):
        context = datapoint['context']['sentences']
        total_context = []

        for topic in context:
            total_context.append("".join(topic))
            context_thing = "".join(total_context)


        prompt_multi = f'<s>\n[INST] <<SYS>>\nYou are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context. The final answer must always be in a new line and should be prepended with "Final Answer: ". The final answer should just be the answer and no explanation\nHere are some examples of past conversations:\n\
Context: ......\nQuestion: Who killed Julius Caesar?\nAnswer: According to the context, Brutus killed Julius Caesar.\nFinal Answer: Brutus\n\n\
Context: ......\nQuestion: Who invented the light bulb?\nAnswer: According to the context, Thomas Edison invented the light bulb.\nFinal Answer: Thomas Edison\n<</SYS>>\n\n\
Context: ......\nQuestion: Who government position does Shirley Temple have?\nAnswer: According to the context, Shirley temple holds the position of Chief of Protocol\nFinal Answer: Chief of Protocol\n<</SYS>>\n\n\
Context: {context_thing}\n\
Question: {datapoint["question"]} [/INST]'
        json_content.append({'prompt': prompt_multi})

        count += 1
    return json_content


def chain_of_thought_prompt_subqa_mixtral(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset, total=len(dataset)):
        context = [i[1] for i in datapoint['context']]
        total_context = []

        for topic in context:
            # print(topic)
            # print("here")
            total_context.append("".join(topic))
            context_thing = "".join(total_context)

        prompt_multi = f"<s> [INST] You are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context.\
                You will break the questions into sub questions. You will then use these sub-questions to get to the final answer. The final answer must be have 'Final Answer: ' prepended to it . Thus your output will be in the following format:\nSub-question 1: [subquestion 1]\nAnswer: [answer 1]\nsub-question 2: [subquestion 2]\nAnswer: [answer 2]\nSub-question n: [subquestion n]\nAnswer: [answer n]\nFinal Answer: [final answer]. The final answer should be limited to 5 words with just the answer and no explanation/information.\n\nHere are some past conversations:\nContext: ......\nQuestion: \'Which government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell\'?\nSub-question 1: Which woman portrayed Corliss Archer in the film Kiss and Tell?.\nAnswer: Shirley Temple.\nSub-question-2 : Which government position was held by Shirley Temple?\nAnswer: Chief of Protocol\nFinal Answer: Chief of Protocol.\n\n\
Context : .......\n\
Question: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?\n\
Sub-question 1: Which university has its main campus in Lawrence, Kansas and has branch campuses in Kansas City metropolitan area?\n\
Answer: University of Kansas\n\
Sub-question 2: What is the name of the fight song of University of Kansas?\n\
Answer: Kansas Song\n\
Final Answer: Kansas Song\n\
\n\n\
Context: {context_thing}\n\
Question: {datapoint['question']} [/INST]"
        json_content.append({'prompt': prompt_multi})
        count += 1
    return json_content


def chain_of_thought_prompt_gpt(dataset):
    json_content = []
    for datapoint in tqdm(dataset, total=len(dataset)):
        context = [i[1] for i in datapoint['context']]
        total_context = []

        for topic in context:
            # print(topic)
            # print("here")
            total_context.append("".join(topic))
            context_thing = "".join(total_context)

        system_prompt = f"You are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context.\
                You will break the questions into sub questions. You will then use these sub-questions to get to the final answer. The final answer must be have 'Final Answer: ' prepended to it . Thus your output will be in the following format:\nSub-question 1: [subquestion 1]\nAnswer: [answer 1]\nsub-question 2: [subquestion 2]\nAnswer: [answer 2]\nSub-question n: [subquestion n]\nAnswer: [answer n]\nFinal Answer: [final answer]. The final answer should be limited to 5 words with just the answer and no explanation/information.\n\nHere are some past conversations:\nContext: ......\nQuestion: \'Which government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell\'?\nSub-question 1: Which woman portrayed Corliss Archer in the film Kiss and Tell?.\nAnswer: Shirley Temple.\nSub-question-2 : Which government position was held by Shirley Temple?\nAnswer: Chief of Protocol\nFinal Answer: Chief of Protocol.\n\n\
Context : .......\n\
Question: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?\n\
Sub-question 1: Which university has its main campus in Lawrence, Kansas and has branch campuses in Kansas City metropolitan area?\n\
Answer: University of Kansas\n\
Sub-question 2: What is the name of the fight song of University of Kansas?\n\
Answer: Kansas Song\n\
Final Answer: Kansas Song\n"
        user_prompt = f"Context: {context_thing}\n\
Question: {datapoint['question']}"

        # json_content.append({'prompt': prompt_multi})
        json_content.append({'system_prompt': system_prompt, 'user_prompt': user_prompt})
    return json_content

def chain_of_thought_prompt_control_gpt(dataset):
    json_content = []
    for datapoint in tqdm(dataset, total=len(dataset)):
        context = [i[1] for i in datapoint['context']]
        total_context = []

        for topic in context:
            # print(topic)
            # print("here")
            total_context.append("".join(topic))
            context_thing = "".join(total_context)


        system_prompt = f'You are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context. The final answer must always be in a new line and should be prepended with "Final Answer: ". The final answer should just be the answer and no explanation\nHere are some examples of past conversations:\n\
Context: ......\nQuestion: Who killed Julius Caesar?\nAnswer: According to the context, Brutus killed Julius Caesar.\nFinal Answer: Brutus\n\n\
Context: ......\nQuestion: Who invented the light bulb?\nAnswer: According to the context, Thomas Edison invented the light bulb.\nFinal Answer: Thomas Edison\n<</SYS>>\n\n\
Context: ......\nQuestion: Who government position does Shirley Temple have?\nAnswer: According to the context, Shirley temple holds the position of Chief of Protocol\nFinal Answer: Chief of Protocol\n<</SYS>>\n\n'
        user_prompt = f"Context: {context_thing}\n\
Question: {datapoint['question']}"

        json_content.append({'system_prompt': system_prompt, 'user_prompt': user_prompt})

    return json_content


def chain_of_thought_prompt_control(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset):
        context = datapoint['context']['sentences']
        total_context = []

        for topic in context:
            total_context.append("".join(topic))
            context_thing = "".join(total_context)


        prompt_multi = f'<s>\n[INST] <<SYS>>\nYou are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context. The final answer must always be in a new line and should be prepended with "Final Answer: ". The final answer should just be the answer and no explanation\nHere are some examples of past conversations:\n\
Context: ......\nQuestion: Who killed Julius Caesar?\nAnswer: According to the context, Brutus killed Julius Caesar.\nFinal Answer: Brutus\n\n\
Context: ......\nQuestion: Who invented the light bulb?\nAnswer: According to the context, Thomas Edison invented the light bulb.\nFinal Answer: Thomas Edison\n<</SYS>>\n\n\
Context: ......\nQuestion: Who government position does Shirley Temple have?\nAnswer: According to the context, Shirley temple holds the position of Chief of Protocol\nFinal Answer: Chief of Protocol\n<</SYS>>\n\n\
Context: {context_thing}\n\
Question: {datapoint["question"]} [/INST]'
        json_content.append({'prompt': prompt_multi})

        count += 1
    return json_content


def chain_of_thought_prompt_control_subqa(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset):
        context = [i[1] for i in datapoint['context']]
        total_context = []

        for topic in context:
            total_context.append("".join(topic))
            context_thing = "".join(total_context)


        prompt_multi = f'<s>\n[INST] <<SYS>>\nYou are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context. The final answer must always be in a new line and should be prepended with "Final Answer: ". The final answer should just be the answer and no explanation\nHere are some examples of past conversations:\n\
Context: ......\nQuestion: Who killed Julius Caesar?\nAnswer: According to the context, Brutus killed Julius Caesar.\nFinal Answer: Brutus\n\n\
Context: ......\nQuestion: Who invented the light bulb?\nAnswer: According to the context, Thomas Edison invented the light bulb.\nFinal Answer: Thomas Edison\n<</SYS>>\n\n\
Context: ......\nQuestion: Who government position does Shirley Temple have?\nAnswer: According to the context, Shirley temple holds the position of Chief of Protocol\nFinal Answer: Chief of Protocol\n<</SYS>>\n\n\
Context: {context_thing}\n\
Question: {datapoint["question"]} [/INST]'
        json_content.append({'prompt': prompt_multi})

        count += 1
    return json_content
# print(actual_chain_of_thought[21]["answer"])

def chain_of_thought_prompt_control_subqa_mixtral(dataset):
    json_content = []
    count = 0
    for datapoint in tqdm(dataset):
        context = [i[1] for i in datapoint['context']]
        total_context = []

        for topic in context:
            total_context.append("".join(topic))
            context_thing = "".join(total_context)


        prompt_multi = f'<s> [INST] You are a helpful, respectful and honest question answering assistant. You will be given a context and a question. Answer the question using only the context. The final answer must always be in a new line and should be prepended with "Final Answer: ". The final answer should just be the answer, no explanation and no more than 5 words.\nHere are some examples of past conversations:\n\
Context: ......\nQuestion: Who killed Julius Caesar?\nAnswer: According to the context, Brutus killed Julius Caesar.\nFinal Answer: Brutus\n\n\
Context: ......\nQuestion: Who invented the light bulb?\nAnswer: According to the context, Thomas Edison invented the light bulb.\nFinal Answer: Thomas Edison\n\n\n\
Context: ......\nQuestion: Who government position does Shirley Temple have?\nAnswer: According to the context, Shirley temple holds the position of Chief of Protocol\nFinal Answer: Chief of Protocol\n\n\
Context: {context_thing}\n\
Question: {datapoint["question"]} [/INST]'
        json_content.append({'prompt': prompt_multi})

        count += 1
    return json_content

if __name__ == "__main__":
    dataset = load_from_disk("hotpotqa.hf")
    validation_dataset = dataset["validation"]
    actual_chain_of_thought = chain_of_thought_prompt(validation_dataset)
    actual_chain_of_thought_control = chain_of_thought_prompt_control(validation_dataset)
    print(actual_chain_of_thought_control[2]["prompt"])
    thing = [i["prompt"] for i in actual_chain_of_thought_control]
    thing1 = [i["prompt"] for i in actual_chain_of_thought]
    print(len(set(thing)))
    print(len(set(thing1)))
    with open('../hotpot/hotpot_chain_of_thought.json', 'w') as fp:
        json.dump(actual_chain_of_thought, fp)

    with open('../hotpot/hotpot_chain_of_thought_control.json', 'w') as fp:
        json.dump(actual_chain_of_thought_control, fp)
