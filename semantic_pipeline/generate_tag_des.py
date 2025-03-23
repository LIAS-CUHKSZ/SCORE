import openai
import json
from tqdm import tqdm
from openai import AzureOpenAI
import argparse
import os

parser = argparse.ArgumentParser(
    description="Generate LLM tag descriptions for RAM++ open-set recognition"
)
parser.add_argument("--openai_api_key", default="sk-xxxxx")
parser.add_argument(
    "--output_file_path",
    help="save path of llm tag descriptions",
    default="datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json",
)


def analyze_tags(tag):
    # Generate LLM tag descriptions

    llm_prompts = [
        f"Provide a clear and concise description of what a(n) {tag} looks like:",
        f"How can you clearly and concisely identify a(n) {tag} by its appearance?",
        f"What does a(n) {tag} look like in a clear and concise description?",
        f"What are the key identifying characteristics of a(n) {tag}:",
        f"Please provide a clear and concise description of the visual characteristics of {tag}:",
    ]

    results = {}
    result_lines = []

    result_lines.append(f"a photo of a {tag}.")

    for llm_prompt in tqdm(llm_prompts):

        # send message
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "assistant", "content": llm_prompt}],
            max_tokens=77,
            temperature=0.99,
            n=10,
            stop=None,
        )

        # parse the response
        for item in response.choices:
            result_lines.append(item.message.content.strip())
        results[tag] = result_lines
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_file",
    )
    args = parser.parse_args()

    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("OPENAI_API_VERSION")

    azure_client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}",
    )

    with open(args.text_file, "r") as f:
        img_tags = f.readlines()

    for i in range(len(img_tags)):
        img_tags[i] = img_tags[i].strip()

    tag_descriptions = []

    for tag in img_tags:
        result = analyze_tags(tag)
        tag_descriptions.append(result)

    with open(args.text_file.replace(".txt", ".json"), "w") as f:
        json.dump(tag_descriptions, f, indent=3)
