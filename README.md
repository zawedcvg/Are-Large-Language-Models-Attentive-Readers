# Are-Large-Language-Models-Attenitve-Readers

## Prerequisites

Install the required packages from the requirements.txt file:
```
pip install -r requirements.txt
```
Put in your OpenAI key in the .env file.

## How to use

Run `pipeline.sh` to run all the preprocessing steps.

Note: For the OpenAI requests with no response, it is put as an empty array in the file. These should be handled by rerunning the requests either using `fill_in_failed_req.py` or something else.

Once `pipeline.sh` is run successfully, use `create_prompts_with_fake_para.py` to create the adversarial datasets.

```
python create_prompts_with_fake_para.py [-h] [-t TYPE] [--related | --no-related] [-c COUNT] [-m MODEL] [--same_type | --no-same_type]
```

The parameters are in accordance to section ... of the paper. The model parameter is for using the prompting type of the 3 models defined. If no model is specified, it creates the dataset in the hotpotqa format in `$FINAL_PROMPTS/hotpotqa_format` with `FINAL_PROMPTS` being defined in .env.
