import openai
import time
import json
import datetime
import asyncio
import aiohttp
import os
import sys
sys.path.insert(1, '..')
from dotenv import load_dotenv
load_dotenv()
DIR = os.getenv("DIR")
OPEN_AI_KEY = os.getenv("OPENAI_KEY")


if sys.argv[1] == "other":
    file_to_read = f"{DIR}/other_fake_para_prompts.json"
    file_to_write = f"{DIR}/fake_other_paragraphs.json"
else:
    file_to_read = f"{DIR}/named_fake_para_prompts.json"
    file_to_write = f"{DIR}/fake_named_paragraphs.json"


from tqdm import tqdm

# Define your OpenAI API key
openai.api_key = OPEN_AI_KEY

with open(file_to_read, 'r') as fp:
    other_para_prompts = json.load(fp)

with open(file_to_write, 'r') as fp:
    all_responses = json.load(fp)

print(len(all_responses))
for i in all_responses:
    if len(i) == 0:
        print("oops")

to_rerun = [[index, prompt] for (index, prompt) in enumerate(other_para_prompts) if len(all_responses[index]) == 0]

new_responses = []
print([i[0] for i in to_rerun])

# exit()
# exit()

async def waiting_code(task):
    try:
        ans = await asyncio.wait_for(task, timeout=30)
        return ans
    except:
        return []

async def main():
    tasks = []
    for i in tqdm([k[1] for k in to_rerun]):
        await asyncio.sleep(1)
        tasks.append(asyncio.create_task(
            openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages = [{"role": "system", "content": i["system_prompt"]}, {"role": "user", "content": i["user_prompt_test"]}, 
                            {"role":"assistant", "content": i["assistant_test"]}, {"role": "user", "content": i["user_prompt"]}],
                )))
    try:
        for coro in tqdm(tasks, total=len(tasks)):
            response = await waiting_code(coro)
            new_responses.append(response)

        return new_responses
    except:
        print("something went wrong lmao")

loop = asyncio.new_event_loop()
ans = loop.run_until_complete(main())
print(len(ans))
for index, i in enumerate(ans):
    ind = to_rerun[index][0]
    all_responses[ind] = i

print(len(all_responses))
with open(file_to_write, 'w') as fp:
    json.dump(all_responses, fp)
