import json
import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.insert(1, '..')
from create_prompts import create_fake_named_entities
DIR = os.getenv("DIR")
with open(f"{DIR}/all_named_entities.json", 'r') as fp:
    named_entities = json.load(fp)

open_ai_thing_single_hop = create_fake_named_entities(named_entities[0])
open_ai_thing_second_hop = create_fake_named_entities(named_entities[1])

with open(f"{DIR}/fake_named_entities_openai_prompts.json", 'w') as fp:
    json.dump([open_ai_thing_single_hop, open_ai_thing_second_hop], fp)
