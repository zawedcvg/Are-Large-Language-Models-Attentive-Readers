import json
from datasets import load_from_disk
from transformers.pipelines.pt_utils import KeyDataset
import random
from tqdm import tqdm

random.seed(42)

def create_system_prompt_fake_named():
    system_prompt_fake_named = "You are a helpful, respectful and honest fake named entity generator. You will be given upto 20 different entity types along with an example of that type. For each of the entity types, generate another named entity different of the same entity type given the named entity. There are a total of 18 different entity types. The different types and their definitions are as given below:\n\
PERSON: People, including fictional\n\
NORP: Nationalities or religious or political groups\n\
FACILITY: Buildings, airports, highways, bridges, etc.\n\
ORGANIZATION: Companies, agencies, institutions, etc.\n\
GPE: Countries, cities, states\n\
LOCATION: Non-GPE locations, mountain ranges, bodies of water\n\
PRODUCT: Vehicles, weapons, foods, etc. (Not services)\n\
EVENT: Named hurricanes, battles, wars, sports events, etc.\n\
WORK OF ART: Titles of books, songs, etc.\n\
LAW: Named documents made into laws\n\
LANGUAGE: Any named language\n\
DATE: Absolute or relative dates or periods\n\
TIME: Times smaller than a day\n\
PERCENT: Percentage (including “%”)\n\
MONEY: Monetary values, including unit\n\
QUANTITY: Measurements, as of weight or distance\n\
ORDINAL: “first”, “second”\n\
CARDINAL: Numerals that do not fall under another type\n\
For each of the provided examples, you will generate one named entity of the same type.\n\nEnsure that your final count of entities is equal to the number of entities in the given prompt. Use indices to help with keeping the count."


user_prompt_fake_named = "1) Type: PERSON Example: Corliss Archer\n\
2) Type: WORK_OF_ART Example: Kiss and Tell\n\
3) Type: WORK_OF_ART Example: \"Big Stone Gap\"\n\
4) Type: DATE Example: 2014\n\
5) Type: NORP Example: South Korean\n\
6) Type: ORG Example: the Lewiston Maineiacs\n\
7) Type: GPE Example: Lawrence\n\
8) Type: GPE Example: Kansas\n\
9) Type: GPE Example: Kansas City\n\
10) Type: PERSON Example: Nicolas Cage\n\
11) Type: PERSON Example: Leoni\n\
12) Type: PERSON Example: David Beckham\n\
13) Type: GPE Example: Vermont\n\
14) Type: ORG Example: Catamounts\n\
15) Type: PERSON Example: Kasper Schmeichel\n\
16) Type: PERSON Example: Alexander Kerensky\n\
17) Type: NORP Example: Bolsheviks\n\
18) Type: CARDINAL Example: Seven\n\
19) Type: NORP Example: Italian\n\
20) Type: NORP Example: Japanese\n\n"

assistant_answer_test = "1) Type: PERSON Fake Answer: Corliss Collins\n\
2) Type: WORK_OF_ART Fake Entity: The Great Gatsby\n\
3) Type: WORK_OF_ART Fake Answer: \"The Love Story\"\n\
4) Type: DATE Fake Answer: 1995\n\
5) Type: NORP Fake Answer: North Korean\n\
6) Type: ORG Fake Answer: the Mavericks\n\
7) Type: GPE Fake Answer: New York\n\
8) Type: GPE Fake Answer: Washington\n\
9) Type: GPE Fake Answer: Basques\n\
10) Type: PERSON Fake Answer: Ethan Wright\n\
11) Type: PERSON Fake Answer: Alex Hart\n\
12) Type: PERSON Fake Answer: Lionel Messi\n\
13) Type: GPE Fake Answer: Seattle\n\
14) Type: ORG Fake Entity: the Bulls\n\
15) Type: PERSON Fake Entity: Alex Winkler\n\
16) Type: PERSON Fake Entity: Vladimir Putin\n\
17) Type: NORP Fake Entity: Mensheviks\n\
18) Type: CARDINAL Fake Entity: Eight\n\
19) Type: NORP Fake Entity: Spanish\n\
20) Type: NORP Fake Entity: Chinese\n\n"




def create_context(dataset):
    json_content = []
    for datapoint in tqdm(dataset):
        context = datapoint['context']
        total_context = []

        for topic in context:
            total_context.append("".join(topic[1]).replace("\"", "\\\""))
            context_thing = "".join(total_context)

        prompt_multi = f"<s>[INST] <<SYS>>\
        You are a helpful, respectful and honest question answering assistant. You always answer in the fewest possible words, and provide only the answer and no explanation.  Use only the following context to answer the questions. Context : {context_thing}\n\
        If the question cannot be answered by the information provided by the context,reply with \"information not in the context\"\
        <</SYS>>\
         {datapoint['question']}[/INST]"
        json_content.append({'question': datapoint['question'], 'context': context_thing, 'prompt': prompt_multi, '_id' : datapoint['_id'], 'answer': datapoint['answer']})

    return json_content

def create_fake_para(dataset):
    json_content = []
    system_prompt = "You are a helpful, respectful and honest fake paragraph generating assistant. You will be given two questions. You will first give a fake answer for the first question. Generate a fake paragraph using the information from the first question and the fake answer generated. The answer and information should not be related to any real life entity.\n\
Use the fake answer generated for the first question to replace all instances of '[answer]' in the second question. Use the newly generated question and generate a fake answer for it. Similar to the first question use the fake answer and the question to generate a fake paragraph.\n\
Generate the two paragraphs as articles under 150 words each. Make the paragraphs sound informative. All the answers and paragraphs must be fake and made up of fake names and fake information. The information/names should not reference any one in real life. Generate exactly one paragraph for each question. Remember to replace all instances of '[answer]' with the answer from the first question and adjust the paragraphs accordingly.\n\
Here is an example of a previous conversation:\n\
User:\n\
Which woman portrayed Corliss Archer in the film Kiss and Tell?\n\
What government position was held by [answer]?\n\
System:\n\
Which woman portrayed Corliss Archer in the film Kiss and Tell?\n\
Fake Answer: Jennifer Delemere\n\
Fake paragraph: .......\n\
What government position was held by [answer]?\n\
Adjusted question: What government position was held by Jennifer Delemere?\n\
Fake answer: Secretary of tourism\n\
Fake paragraph: ....."
    for subq1, subq2 in tqdm(dataset):
        prompt_multi = f"<s>[INST] <<SYS>>\n{system_prompt}\n\
<</SYS>>\n\
{subq1}\n{subq2}\n[/INST]"
        json_content.append({'prompt': prompt_multi})

    return json_content

def create_fake_para_openai(dataset):
    json_content = []

    with open("data/dev_ori.json", "r") as fp:
        original_question = json.load(fp)
    with open("data/dev_sub1.json", "r") as fp:
        first_subquestion = json.load(fp)
    system_prompt = "You are a helpful and respectful fake paragraph generating assistant. You will be given two questions, few supporting paragraphs and two words you need to avoid. You will first give a fake answer for the first question. The fake answer should not be the same as any of the two words that needs to be avoided. Generate a fake paragraph using the information from the first question and the fake answer generated. The answer and information should not be related to any real life entity. The paragraphs generated must match the tone of the given two paragraphs. Further, the two paragraphs generated must not contradict any of the information in the supporting paragraphs provided by the user.\n\
Use the fake answer generated for the first question to replace all instances of '[answer]' in the second question. Use the newly generated question and generate a fake answer for it. Ensure that the fake answer generated is not the same as the any of the provided words you need to avoid. Similar to the first question use the fake answer and the question to generate a fake paragraph. You will generate the fake paragraphs as if it were part of a wikipedia article. You must maintain a neutral and informative tone.\n\
Generate the two paragraphs as separate articles about 75-100 words each. All the answers and paragraphs must be made up of fake names and fake information. The information/names should not reference any one in real life. Generate exactly one paragraph for each question. Remember to replace all instances of '[answer]' with the answer from the first question and adjust the paragraphs accordingly. However you must not mention the fact that the details/entities in the paragraphs are fake/imaginary.\n"
    user_prompt_test = "Question 1: Which woman portrayed Corbin Archer in the film Kiss and Tell?\nQuestion 2: What government position was held by [answer]?\nSupporting facts:\n\
Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\n\
Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Collins and directed by Michael Scott. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves."
    assistant_test = "Fake Answer 1: Samantha Lake\nFake paragraph 1: Samantha Lake took on the role of Corbin Archer in the film 'Kiss and Tell,' a charming depiction of youthful romance and mischievous antics. Directed by Michael Scott, the movie captured the essence of teenage life in the 1950s. Samantha Lake's portrayal of Corbin Archer was widely acclaimed for its blend of innocence and wit. Lake's performance contributed to the film's enduring popularity and critical acclaim.\n\nAdjusted question 2: What government position was held by Samantha Lake?\nFake Answer 2: Deputy Secretary of Commerce.\nFake paragraph 2: Samantha Lake held the position of Deputy Secretary of Commerce in the government, where she played a significant role in the formulation of crucial economic initiatives and trade policies."
    for subq1, subq2, idx in tqdm(dataset):
        # req_para = [context[1] for context in original_question[idx]["supporting_facts"]]
        user_prompt = f"{subq1}\n{subq2}\nSupporting facts:\n"
        titles = [context[0] for context in original_question[idx]["context"]]
        first_answer = first_subquestion[idx]["answer"]
        answer = original_question[idx]["answer"]
        req_titles = [context[0] for context in original_question[idx]["supporting_facts"]]
        req_titles_indices = set([index for index, value in enumerate(titles) if value in req_titles])
        req_titles_para = [" ".join(original_question[idx]["context"][index][1]) for index in req_titles_indices]
        for para in req_titles_para:
            user_prompt += f"{para}\n"
        user_prompt += f"Word to avoid: 1) {answer}\n2) {first_answer}"
        json_content.append({'system_prompt': system_prompt, 'user_prompt': user_prompt, 'user_prompt_test': user_prompt_test, 'assistant_test': assistant_test})

    return json_content

def create_fake_named_entities(dataset):
    json_content = []
    system_prompt = "You are a helpful, respectful and honest fake named entity generator. You will be given upto 20 different entity types along with an example of that type. For each of the entity types, generate another named entity different of the same entity type given the named entity. There are a total of 18 different entity types. The different types and their definitions are as given below:\n\
PERSON: People, including fictional\n\
NORP: Nationalities or religious or political groups\n\
FACILITY: Buildings, airports, highways, bridges, etc.\n\
ORGANIZATION: Companies, agencies, institutions, etc.\n\
GPE: Countries, cities, states\n\
LOCATION: Non-GPE locations, mountain ranges, bodies of water\n\
PRODUCT: Vehicles, weapons, foods, etc. (Not services)\n\
EVENT: Named hurricanes, battles, wars, sports events, etc.\n\
WORK OF ART: Titles of books, songs, etc.\n\
LAW: Named documents made into laws\n\
LANGUAGE: Any named language\n\
DATE: Absolute or relative dates or periods\n\
TIME: Times smaller than a day\n\
PERCENT: Percentage (including “%”)\n\
MONEY: Monetary values, including unit\n\
QUANTITY: Measurements, as of weight or distance\n\
ORDINAL: “first”, “second”\n\
CARDINAL: Numerals that do not fall under another type\n\
For each of the provided examples, you will generate one named entity of the same type.\n\nEnsure that your final count of entities is equal to the number of entities in the given prompt. Use indices to help with keeping the count."
    user_prompt_test = "1) Type: PERSON Example: Corliss Archer\n\
2) Type: WORK_OF_ART Example: Kiss and Tell\n\
3) Type: WORK_OF_ART Example: \"Big Stone Gap\"\n\
4) Type: DATE Example: 2014\n\
5) Type: NORP Example: South Korean\n\
6) Type: ORG Example: the Lewiston Maineiacs\n\
7) Type: GPE Example: Lawrence\n\
8) Type: GPE Example: Kansas\n\
9) Type: GPE Example: Kansas City\n\
10) Type: PERSON Example: Nicolas Cage\n\
11) Type: PERSON Example: Leoni\n\
12) Type: PERSON Example: David Beckham\n\
13) Type: GPE Example: Vermont\n\
14) Type: ORG Example: Catamounts\n\
15) Type: PERSON Example: Kasper Schmeichel\n\
16) Type: PERSON Example: Alexander Kerensky\n\
17) Type: NORP Example: Bolsheviks\n\
18) Type: CARDINAL Example: Seven\n\
19) Type: NORP Example: Italian\n\
20) Type: NORP Example: Japanese\n\n"

    assistant_answer_test = "1) Type: PERSON Fake Answer: Corliss Collins\n\
2) Type: WORK_OF_ART Fake Entity: The Great Gatsby\n\
3) Type: WORK_OF_ART Fake Answer: \"The Love Story\"\n\
4) Type: DATE Fake Answer: 1995\n\
5) Type: NORP Fake Answer: North Korean\n\
6) Type: ORG Fake Answer: the Mavericks\n\
7) Type: GPE Fake Answer: New York\n\
8) Type: GPE Fake Answer: Washington\n\
9) Type: GPE Fake Answer: Basques\n\
10) Type: PERSON Fake Answer: Ethan Wright\n\
11) Type: PERSON Fake Answer: Alex Hart\n\
12) Type: PERSON Fake Answer: Lionel Messi\n\
13) Type: GPE Fake Answer: Seattle\n\
14) Type: ORG Fake Entity: the Bulls\n\
15) Type: PERSON Fake Entity: Alex Winkler\n\
16) Type: PERSON Fake Entity: Vladimir Putin\n\
17) Type: NORP Fake Entity: Mensheviks\n\
18) Type: CARDINAL Fake Entity: Eight\n\
19) Type: NORP Fake Entity: Spanish\n\
20) Type: NORP Fake Entity: Chinese\n\n"


    count = 0
    total_count = 0
    temp_thing = ""
    for name_entities in tqdm(dataset):
        for i in name_entities:
            total_count += 1
            if count == 19:
                temp_thing += f"{count + 1}) Type: {i[3]} Example: {i[0]}\n"
                json_content.append({'system_prompt': system_prompt, 'user_prompt': temp_thing, 'assistant_test': assistant_answer_test, 'user_prompt_test': user_prompt_test})
                count = 0
                temp_thing = ""
            else:
                temp_thing += f"{count + 1}) Type: {i[3]} Example: {i[0]}\n"
                count += 1

    if count != 0:
        json_content.append({'system_prompt': system_prompt, 'user_prompt': temp_thing, 'assistant_test': assistant_answer_test, 'user_prompt_test': user_prompt_test})

    return json_content

if __name__ == "main":
    with open('../subqa/dev_ori.json', 'r') as fp:
        multi_hop = json.load(fp)

    with open('../subqa/dev_sub1.json', 'r') as fp:
        single_hop = json.load(fp)

    with open('../subqa/dev_sub2.json', 'r') as fp:
        second_hop = json.load(fp)


    multi_json_content = create_context(multi_hop)

    with open('../subqa/prompt_multi.json', 'w') as fp:
        json.dump(multi_json_content, fp)

    single_json_content = create_context(single_hop)

    second_json_content = create_context(second_hop)

    with open('../subqa/prompt_single.json', 'w') as fp:
        json.dump(single_json_content, fp)

    with open('../subqa/prompt_second.json', 'w') as fp:
        json.dump(second_json_content, fp)
