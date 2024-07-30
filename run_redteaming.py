import argparse
import json
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from automated_redteaming import *


load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
openai_api_key = os.environ["OPENAI_API_KEY"]

def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file_name", type=str, help="Name of the file to process")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the model to use (default: default_model)")
    parser.add_argument("--device", type=str, default="cuda", help="Name of device")
    parser.add_argument("--attack_type", type=str, default="pair", choices=["pair", "gcg"], help="Name of attack")
    
    return parser


def main(args):
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_access_token,
        torch_dtype=torch.bfloat16
    ).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dataset = load_dataset("abhayesian/augmented_advbench_v5")["train"].shuffle()

    if args.attack_type == "pair":
        pair = PromptAutomaticIterativeRefinement(
            model=model,
            tokenizer=tokenizer,
            api_key=openai_api_key,
        )
    elif args.attack_type == "gcg":
        gcg = GreedyCoordinateGradient(
            model=model,
            tokenizer=tokenizer,
        )

    if os.path.exists(args.file_name):
        with open(args.file_name, "r") as f:
            results = json.load(f)
    else:
        results = []

    for i in range(len(dataset)):
        try:
            prompt = dataset[i]["prompt"]
            target = " ".join(dataset[i]["rejected"].split()[:10])
            chosen_response = dataset[i]["chosen"]
            
            if args.attack_type == "pair":
                # Get PAIR attack
                top_candidate, high_score, response = pair.compute_adversary(prompt, target)
                attack_dict = {
                    "prompt": prompt,
                    "refusal": chosen_response,
                    "jailbreak": top_candidate,
                    "model_response": response,
                    "pair_score": high_score,
                }
            elif args.attack_type == "gcg":
                top_candidate, gcg_loss, response = gcg.compute_adversary(prompt, target)
                attack_dict = {
                    "prompt": prompt,
                    "refusal": chosen_response,
                    "jailbreak": top_candidate,
                    "model_response": response,
                    "gcg_loss": gcg_loss,
                }
            elif args.attack_type == "msj":
                pass

            # Get StrongReject response
            score = autograde_response(prompt, response).score_zero_to_one
            if score == 0: # If refusal
                continue
            attack_dict["strongreject_score"] = score

            # Add new example to the file
            results.append(attack_dict)            
            with open(args.file_name, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            raise e


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)