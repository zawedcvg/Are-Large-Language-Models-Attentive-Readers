# Are-Large-Language-Models-Attentive-Readers

This project is a framework to create adversarial paragraphs for the HotpotQA dataset, by treating each question in HotpotQA as a two-hop question . It extracts relevant information from each of these hops using [Stanza](https://github.com/stanfordnlp/stanza), and uses several techniques to fake sub-questions which are then fed into GPT-4 to create the fake paragraphs.

Refer to this paper: [link to the paper]

## Prerequisites

- Python 3.10.12

Install the required packages from the `requirements.txt` file:

```
pip install -r requirements.txt
```

Put your OpenAI API key in the `.env` file.

## How to Use

Run `pipeline.sh` to execute all the preprocessing steps.

**Note:** For OpenAI requests with no response, they are stored as an empty array in the file. These should be handled by rerunning the requests either using `fill_in_failed_req.py` or another script.

Once `pipeline.sh` is run successfully, use `create_prompts_with_fake_para.py` to create the adversarial datasets.

```
python create_prompts_with_fake_para.py [-h] [-t TYPE] [--related | --no-related] [-c COUNT] [-m MODEL] [--same_type | --no-same_type]
```

The parameters correspond to the section in the paper. The `MODEL` parameter is for using the prompting type of the three models defined. If no model is specified, it creates the dataset in the HotpotQA format in `$FINAL_PROMPTS/hotpotqa_format`, where `FINAL_PROMPTS` is defined in the `.env` file.

## The Pipeline

### Extracting the Modifiable Parts

`getting_modifiable_parts.py` parses the sub-questions to find the main object and modifiable portions (converted to "other" and "named-entities"). To pass in your own sub-questions, you need to modify this file.

### Creating the Fake Sub-questions

`create_prompts_for_fake_named.py` creates OpenAI prompts to generate fake named entities to replace the named-entities extracted from the previous step. `create_fake_entities.py` uses the prompts and sends requests to OpenAI, and `extract_fake_named_entities.py` extracts the fake named entities from the OpenAI responses.

`create_real_tuples` is a preprocessing step where tuples are created from the sub-questions. `create_fake_questions.py` uses the modifiable portions from the first step along with the fake named-entities extracted to create fake questions, which will later be used to create the adversarial paragraphs.

### Creating the Adversarial Paragraphs

`create_fake_question_tuples.py` combines different fake sub-questions with the same base question to generate multiple fake sub-question pairs. This script also creates OpenAI prompts to create adversarial paragraphs based on these.

`create_fake_para.py` sends the requests, and `extract_fake_paragraphs.py` extracts the fake/adversarial paragraphs.
