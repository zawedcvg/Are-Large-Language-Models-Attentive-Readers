import openai
import time
import json
import datetime
import asyncio
import sys
import aiohttp
import os
from dotenv import load_dotenv
load_dotenv()

DIR = os.getenv("DIR")
from tqdm import tqdm

# Define your OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")


def backoff_delay(backoff_factor, attempts):
    # backoff algorithm
    delay = backoff_factor * (2 ** attempts)
    return delay
def retry_request():
    pass


# with open("./other_fake_para_prompts.json", 'r') as fp:
    # other_para_prompts = json.load(fp)

# file_to_write = "fake_other_para.json"

if sys.argv[1] == "other_prompts":
    file_to_read = f"{DIR}/other_fake_para_prompts.json"
    file_to_write = f"{DIR}/fake_other_paragraphs.json"
else:
    file_to_read = f"{DIR}/named_fake_para_prompts.json"
    file_to_write = f"{DIR}/fake_named_paragraphs.json"


# for i in file_to_read:
    # print(i)

# exit()

with open(f"{file_to_read}", 'r') as fp:
    named_para_prompts = json.load(fp)

# # for i in named_para_prompts:
    # # print(i)
# print(len(named_para_prompts))
# exit()

# file_to_write = f"{DIR}/fake_other_para_test_1.json"

all_responses = []

async def waiting_code(task, tries):
    try:
        ans = await asyncio.wait_for(task, timeout=15)
        return ans
    except:
        tries += 1
        if tries < 4:
            delay = 2 * (2 ** tries)
            time.sleep(delay)
            ans = await waiting_code(task, tries)
            return ans
        else:
            print("This failed you suck")
            return []

async def main():
    tasks = []
    for i in tqdm(named_para_prompts):
        await asyncio.sleep(1)
        tasks.append(asyncio.create_task(
            openai.ChatCompletion.acreate(
                model="gpt-4-turbo-preview",
                messages = [{"role": "system", "content": i["system_prompt"]}, {"role": "user", "content": i["user_prompt_test"]},
                            {"role":"assistant", "content": i["assistant_test"]}, {"role": "user", "content": i["user_prompt"]}],
                )))
    try:
        for coro in tqdm(tasks, total=len(tasks)):
            response = await waiting_code(coro, 0)
            all_responses.append(response)
        return all_responses
    except:
        print("something went wrong lmao")

loop = asyncio.new_event_loop()
ans = loop.run_until_complete(main())
print(len(ans))
with open(file_to_write, 'w') as fp:
    json.dump(ans, fp)
