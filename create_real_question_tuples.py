import json
import os
from dotenv import load_dotenv
load_dotenv()

DIR = os.getenv("DIR")

with open('../../subqa/dev_ori.json', 'r') as fp:
    multi_hop = json.load(fp)

with open('../../subqa/dev_sub1.json', 'r') as fp:
    single_hop = json.load(fp)

with open('../../subqa/dev_sub2.json', 'r') as fp:
    second_hop = json.load(fp)

json_thing = []

for first, second in zip(single_hop, second_hop):
    first_question = first["question"]
    second_question = second["question"]
    answer = first["answer"]
    json_thing.append([first_question, second_question, answer])

with open(f'{DIR}/subqa_tuples.json', 'w') as fp:
    json.dump(json_thing, fp)
