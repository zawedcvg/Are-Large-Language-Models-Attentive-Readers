import json
import os
from dotenv import load_dotenv
load_dotenv()

DIR = os.getenv("DIR")
# with open(f'{DIR}/fake_entities_generated.json') as fp:
    # all_responses = json.load(fp)
with open(f'{DIR}/fake_entities_generated.json') as fp:
    all_responses = json.load(fp)

with open(f'{DIR}/fake_named_entities_openai_prompts.json') as fp:
    all_things = json.load(fp)



print(len(all_responses))
print(len(all_things[0]))
print(len(all_responses[0]))
input()
all_named_entities = [[], []]
for thing, i in zip(all_things[0], all_responses[0]):
    to_go_through = i["choices"][0]["message"]["content"].split("\n")
    for named_ents in to_go_through:
        print(len(to_go_through))
        if len(to_go_through) < 20:
            print(thing["user_prompt"])
        all_named_entities[0].append(named_ents[named_ents.rfind(":") + 1:].strip())

for thing, i in zip(all_things[1], all_responses[1]):
    to_go_through = i["choices"][0]["message"]["content"].split("\n")
    for named_ents in to_go_through:
        print(len(to_go_through))
        if len(to_go_through) < 20:
            print(thing["user_prompt"])
        all_named_entities[1].append(named_ents[named_ents.rfind(":") + 1:].strip())


print(len(all_named_entities[0]))
print(len(all_named_entities[1]))

with open(f'{DIR}/all_fake_named_entities_extracted.json', 'w') as fp:
    json.dump(all_named_entities, fp)

