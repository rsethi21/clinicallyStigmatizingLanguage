import os
import yaml
import json
import argparse
from tqdm import tqdm
import pdb

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np

# arguments for commandline interface
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--configurations_fp", help="path to configurations file for RAG experiment", required=True) # configs for the experiment
parser.add_argument("-d", "--notes_fp", help="path to file with clinical notes in csv format", required=True) # input csv with clinical notes
parser.add_argument("-o", "--output_fp", help="path to folder for outputs", required=True) # output folder path

# load yaml configuration file
def readYaml(fp: str):
    with open(fp, "r") as file:
        dictionary = yaml.safe_load(file)
    return dictionary

# generate embedding vectors for text using a pipeline object in transformers
def embeddings(i: str, pipeline_ee: object):
    return pipeline_ee(i)

# calculate a perplexity score for text
def perplexity(str1, str2, model, tokenizer):
  inputs = tokenizer(str1 + "\n" + str2, return_tensors="pt")
  loss = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
  ppl = torch.exp(loss)
  return ppl

# calculuate dot product
def mean_dot_similarity(str1: str, str2: str):
    return np.dot(str1, np.transpose(str2))

# calculate cosine similarity
def mean_cosine_similarity(str1: str, str2: str):
    return np.dot(str1, np.transpose(str2))/(np.linalg.norm(str1)*np.linalg.norm(str2))

# function to select sequences that are most similar
def similarity_selection(i, sequences_to_compare, pipe, sim_fn=mean_cosine_similarity, num=1, model=None, tokenizer=None, threshold=None):
    # convert string to numerical vectors (average token embeddings)
    if sim_fn != perplexity:
        str1_embeddings = embeddings(i, pipe)
        str1_vector = np.mean(np.array(str1_embeddings[0]), axis=0)
    # make comparisons between sequence i and the sequences to compare
    scores = []
    for comparison in tqdm(sequences_to_compare, desc="RAG Selection"):
        if sim_fn != perplexity:
            # extract sequence to compare embedding
            str2_embeddings = embeddings(comparison, pipe)
            str2_vector = np.mean(np.array(str2_embeddings[0]), axis=0)
            # calculate similarity with the similarity vector of the user's choice
            scores.append(sim_fn(str1_vector, str2_vector))
        # otherwise make comparison with perplexity function that does not require embeddings
        else:
            scores.append(sim_fn(comparison, i, model, tokenizer))
    # determine the most likely entries
    sorted_scores = sorted(scores)
    # select top n context entries, none if none of them meet threshold
    if threshold != None and sim_fn == mean_cosine_similarity:
        if sorted_scores[-1] < threshold:
            selected_context = None
        else:
            selected_context = ""
            for n in range(num):
                find_score = sorted_scores[-1*(n+1)]
                ind = scores.index(find_score)
                selected_context += f"{sequences_to_compare[ind]}\n\n"
    # select top n context entries
    else:
        selected_context = ""
        for n in range(num):
            find_score = sorted_scores[-1*(n+1)]
            ind = scores.index(find_score)
            selected_context += f"{sequences_to_compare[ind]}\n\n"
    return selected_context

# format prompt for LLM
def format_message(sp: str, input: str, context: str or None = None, model_v: str = "llama3"):
    """
    Purpose: create chat dialog following appropriate format.
    Input: system prompt, context, input, model version
    Output: formatted prompt
    """
    # asserting only certain formatting for chat prompt
    assert model_v in ["llama2", "llama3", "none"], "Only formatting allowed: llama2, llama3, none (which is llama2)."
    # RAG context string
    if context != None:
        context_priming = f"Please use the context provided below to answer the question.\nContext:\n{context}"
    else:
        context_priming = ""
    # formatting llama2/mistral/mixtral prompt
    if model_v == "llama2" or model_v == "none":
        start = "[INST]\n" # start token
        system_prompt = f"<<SYS>>\n{sp}\n<</SYS>>\n" # question/task
        input_data = f"Below is the clinical note to use for the question:\n{input}\n" + context_priming # user input
        end = "[/INST]\n\n" # end token
    # formatting llama3 prompt
    elif model_v == "llama3":
        start = "<|begin_of_text|>\n" # start token
        system_prompt = f"<|start_header_id|>system<|end_header_id|>{sp}\n<|eot_id|>\n" # question/task
        input_data = f"<|start_header_id|>user<|end_header_id|>\nBelow is the clinical note to use for the question:\n{input}\n{context_priming}<|eot_id|>\n" # user input
        end = "<|start_header_id|>assistant<|end_header_id|>" # end token
    return start+system_prompt+input_data+end

# run an inference instance for an LLM
def generate(prompt: str, pipeline: object, tokenizer: object, parameters: dict):
    # asserting only certain formatting for output parsing purposes
    assert parameters["method"]["formatting"] in ["llama2", "llama3", "none"], "Only formatting allowed: llama2, llama3, none (which is llama2)."
    # generating output
    generated_outputs = pipeline(
        prompt,
        **parameters["llm"])
    # parsing llama3 output
    if parameters["method"]["formatting"] == "llama3":
        start_index = generated_outputs[0]['generated_text'].rindex('<|start_header_id|>assistant<|end_header_id|>') + len("<|start_header_id|>assistant<|end_header_id|>")
        generated_text = generated_outputs[0]['generated_text'][start_index:]
        generated_text = generated_text.replace("\n", "")
        generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
    # parsing llama2, mistral/mixtral output
    else:
        end_index = generated_outputs[0]['generated_text'].index('[/INST]')
        generated_text = generated_outputs[0]['generated_text'][end_index+len("[/INST]"):]
        generated_text = generated_text.replace("\n", "")
        generated_text = generated_text.encode('ascii', errors='ignore').strip().decode('ascii')
    return generated_text

# reprompting technique; basically asking the model to re-evaluate itself based on its original answer
def query_transform(prompt: str, pipeline: object, tokenizer: object, parameters: dict):
    appendation = "Here is your original analysis of stigmatizing language in the clinical note: "
    initial_answer = generate(prompt, pipeline, tokenizer, parameters)
    return "\n" + appendation + initial_answer + "\n" + "You may borrow from or change this analysis with the given stigmatizing language context/guidance below." + "\n"

# chunking function
def create_sub_queries(query, chunk_size, chunk_overlap, tokenizer):
    assert chunk_size > chunk_overlap, "Overlap must be less than size of chunk in tokens"
    tokens = tokenizer(query, return_tensors="pt")
    tokens = tokens["input_ids"].numpy()[0]
    sub_queries = []
    for i in range(0, len(tokens)-chunk_overlap, chunk_size-chunk_overlap):
        if i == list(range(0, len(tokens)-chunk_overlap, chunk_size-chunk_overlap))[-1]:
            sub_queries.append(tokenizer.decode(np.array(tokens[i:])))
        else:
            sub_queries.append(tokenizer.decode(np.array(tokens[i:i+chunk_size])))
    return sub_queries


if __name__ == "__main__":
    # arguments
    args = parser.parse_args()

    # parameters
    parameters = readYaml(args.configurations_fp)

    similarity_mapping = {"dot": mean_dot_similarity, "cos": mean_cosine_similarity, "perplexity": perplexity}

    # instantiating model and pipeline
    tokenizer = AutoTokenizer.from_pretrained(parameters["method"]["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        parameters["method"]["model_path"],
        device_map='auto',
        local_files_only=True)
    pipeline_tg = transformers.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map='auto')
    pipeline_ee = transformers.pipeline(
        'feature-extraction',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map='auto')
    
    # openning csv file
    input_df = pd.read_csv(args.notes_fp)
    input_notes = list(input_df["Assessment"])

    # openning context file
    if parameters["method"]["context_path"] != None:
        input_context = pd.read_csv(parameters["method"]["context_path"])
        input_context_texts = list(input_context["Context"])

    outputs = []
    selected_contexts = []
    # iterate notes and apply RAG
    for i in tqdm(input_notes, desc="Notes"):
        # if there is no chunking
        if parameters["method"]["chunking"] == None:
            # if there is a context entry rag dataset entered
            if parameters["method"]["context_path"] != None:
                # if there is requirement for sequential prompting
                if parameters["method"]["transform"]:
                    basic_prompt = format_message(parameters["method"]["sp"], i, context=None, model_v=parameters["method"]["formatting"])
                    i = i + query_transform(basic_prompt, pipeline_tg, tokenizer, parameters)
                # if there is set number of context entries, use top n using similarity selection (RAG)
                if parameters["method"]["num_context"] != None:
                    selected_context = similarity_selection(i, input_context_texts, pipeline_ee, sim_fn=similarity_mapping[parameters["method"]["scoring"]], num=parameters["method"]["num_context"], model=model, tokenizer=tokenizer, threshold=parameters["method"]["threshold"])
                # otherwise append all context entries (in-context approach)
                else:
                    selected_context = "\n\n".join(input_context_texts)
            # otherwise, no context augmentation
            else:
                selected_context = None
            # take selected contexts and format prompt as such, then generate response using context if selected
            selected_contexts.append(selected_context)
            prompt = format_message(parameters["method"]["sp"], i, context=selected_context, model_v=parameters["method"]["formatting"])
            outputs.append(generate(prompt, pipeline_tg, tokenizer, parameters))
        # if there is chunking, essentially the same thing as above except applied to each chunk separately
        else:
            temp_contexts = []
            temp_outputs = []
            # separate note into chunks and repeat the above for each chunk
            note_chunks = create_sub_queries(i, int(parameters["method"]["chunking"]["chunk_size"]), int(parameters["method"]["chunking"]["overlap_size"]), tokenizer)
            for ind, sub_i in tqdm(enumerate(note_chunks), desc="Note Chunk", total=len(note_chunks)):
                if parameters["method"]["context_path"] != None:
                    if parameters["method"]["transform"]:
                        basic_prompt = format_message(parameters["method"]["sp"], sub_i, context=None, model_v=parameters["method"]["formatting"])
                        sub_i = sub_i + query_transform(basic_prompt, pipeline_tg, tokenizer, parameters)
                    if parameters["method"]["num_context"] != None:
                        selected_context = similarity_selection(sub_i, input_context_texts, pipeline_ee, sim_fn=similarity_mapping[parameters["method"]["scoring"]], num=parameters["method"]["num_context"], model=model, tokenizer=tokenizer, threshold=parameters["method"]["threshold"])
                    else:
                        selected_context = "\n\n".join(input_context_texts)
                else:
                    selected_context = None
                temp_contexts.append(f"Note Chunk {ind+1} Context: {selected_context}")
                prompt = format_message(parameters["method"]["sp"], sub_i, context=selected_context, model_v=parameters["method"]["formatting"])
                temp_outputs.append(generate(prompt, pipeline_tg, tokenizer, parameters))
            selected_contexts.append("\n\n".join(temp_contexts))
            outputs.append(temp_outputs)
    
    # save output as jsons and the asscoiated selected contexts for later error analysis
    with open(os.path.join(args.output_fp, "generated_outputs.json"), "w") as outfile_generation:
        json.dump(outputs, outfile_generation)
    with open(os.path.join(args.output_fp, "contexts.json"), "w") as outfile_context:
        json.dump(selected_contexts, outfile_context)