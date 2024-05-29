# using llama-cpp-python to do (eventually) synthetic data generation via instructor

import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3-8B-Instruct-Q6_K.gguf"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# For this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# like Mistral 7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.
#
# Tip: avoid using global variables in this function.
# Changes to code outside this function will not be detected, and the download step will not re-run.
def download_model_to_image(model_dir, model_name, model_file):
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    hf_hub_download(
        repo_id=model_name,
        filename=model_file,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


# ### Image definition
# and on the first day, god created cuda
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "transformers",
        "flash-attention",
    )
    .env({"CMAKE_ARGS": "-DLLAMA_CUDA=on"})
    .pip_install(
        "llama-cpp-python",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME, "model_file": MODEL_FILE},
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App(
    "example-vllm-inference", image=image
)

# Using `image.imports` allows us to have a reference to llama_cpp in global scope without getting an error when our script executes locally.
with image.imports():
    import llama_cpp

# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions),
# using a `load_model` method decorated with `@modal.enter`. This enables us to load the model into memory just once,
# every time a container starts up, and to keep it cached on the GPU for subsequent invocations of the function.
GPU_CONFIG = modal.gpu.A100(count=1)

@app.cls(gpu=GPU_CONFIG, timeout=600)
class Model:
    @modal.enter()
    def load_model(self):
        self.llama = llama_cpp.Llama(
            model_path=f"{MODEL_DIR}/{MODEL_FILE}",
            n_gpu_layers=-1,
            chat_format="llama-3",
            n_ctx=2048,
            logits_all=False,
            verbose=False,
            flash_attn=True,
        )

    @modal.method()
    def generate(self, user_questions):
        start = time.monotonic_ns()
        ret = []
        total_tokens = 0
        for prompt in user_questions:
            resp = self.llama.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": prompt,
                }]
            )
            ret.append([prompt, resp['choices'][0]['message']['content']])
            total_tokens += resp['usage']['total_tokens']

        duration_s = (time.monotonic_ns() - start) / 1e9

        return {
            "answers": ret,
            "tokens": total_tokens,
            "duration": duration_s,
        }

def split_list(lst, num_parts):
    length = len(lst)
    return [lst[i*length // num_parts: (i+1)*length // num_parts]
            for i in range(num_parts)]

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@app.local_entrypoint()
def main():
    questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    ]
    parts = split_list(questions, 10)

    model = Model()

    total_tokens = 0
    duration = 0
    # this map shit is extremely cool
    for resp in model.generate.map(parts):
        total_tokens += resp["tokens"]
        duration += resp["duration"]
        for r in resp["answers"]:
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{r[0]}",
                f"\n{COLOR['BLUE']}{r[1]}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            print("----------------------------------------------------------")
    print(f"total tokens: {total_tokens}")
    print(f"duration: {duration} seconds")
