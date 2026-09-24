"""
NOVA AI — Language Model
========================

Nova Musical's own decoder-only Transformer language model.

This is NOT a list of programmed answers.

Nova learns statistical language patterns from training text
and generates responses token-by-token.

Architecture:
    Token embeddings
    Positional embeddings
    Causal self-attention
    Transformer blocks
    Feed-forward networks
    Layer normalization
    Language-model output head

The model weights are trained and saved by Nova.
"""

import math
import os
import json
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


CONFIG = {
    "model_name": "Nova AI",
    "version": "0.2-language",

    # Small development model.
    # We can scale these later.
    "context_length": 256,
    "embedding_size": 256,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,

    "batch_size": 16,
    "learning_rate": 3e-4,
    "training_steps": 5000,

    "minimum_token_frequency": 1
}


print()
print("====================================")
print("          NOVA AI TRAINING")
print("====================================")
print()
print("Device:", DEVICE)
print()


# ============================================================
# TRAINING DATA
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "training.txt"
)


if not os.path.exists(DATA_FILE):

    starter = """<user>Hello</user>
<assistant>Hello! I'm Nova. What would you like to work on today?</assistant>

<user>I'm a beginner.</user>
<assistant>Great. We can start from the beginning and take it one step at a time.</assistant>

<user>What is Em?</user>
<assistant>Em means E minor. On guitar, it is one of the simplest open chords to learn.</assistant>

<user>How do I play it?</user>
<assistant>Place two fingers on the second fret, one on the A string and one on the D string, then strum all six strings.</assistant>

<user>idk</user>
<assistant>That's okay. Tell me what part you're unsure about and I'll explain it differently.</assistant>
"""

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(starter)


    print(
        "Created nova-ai/training.txt"
    )


with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    TRAINING_TEXT = file.read()


if len(TRAINING_TEXT.strip()) < 20:

    raise ValueError(
        "training.txt does not contain enough text."
    )


print(
    "Training characters:",
    len(TRAINING_TEXT)
)


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_PATTERN = re.compile(
    r"""
    <user>|
    </user>|
    <assistant>|
    </assistant>|
    [A-Za-z]+(?:'[A-Za-z]+)?|
    [0-9]+|
    [^\w\s]
    """,
    re.VERBOSE
)


def tokenize(text):

    return TOKEN_PATTERN.findall(text)


tokens = tokenize(
    TRAINING_TEXT
)


counter = Counter(tokens)


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>"
]


vocabulary = list(
    SPECIAL_TOKENS
)


for token, frequency in sorted(
    counter.items()
):

    if (
        frequency >=
        CONFIG["minimum_token_frequency"]
        and
        token not in vocabulary
    ):

        vocabulary.append(token)


stoi = {
    token: index
    for index, token
    in enumerate(vocabulary)
}


itos = {
    index: token
    for token, index
    in stoi.items()
}


VOCAB_SIZE = len(vocabulary)


CONFIG["vocab_size"] = VOCAB_SIZE


print(
    "Vocabulary size:",
    VOCAB_SIZE
)


def encode(text):

    result = []

    for token in tokenize(text):

        result.append(
            stoi.get(
                token,
                stoi["<unk>"]
            )
        )

    return result


def decode(ids):

    pieces = []

    punctuation = {
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        ")",
        "]"
    }

    opening = {
        "(",
        "["
    }

    for token_id in ids:

        token = itos.get(
            int(token_id),
            "<unk>"
        )

        if token in SPECIAL_TOKENS:
            continue

        if not pieces:

            pieces.append(token)

        elif token in punctuation:

            pieces[-1] += token

        elif pieces[-1] in opening:

            pieces[-1] += token

        else:

            pieces.append(
                " " + token
            )

    return "".join(pieces)


encoded_training_data = torch.tensor(
    encode(TRAINING_TEXT),
    dtype=torch.long
)


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(self):

        super().__init__()

        size = CONFIG[
            "embedding_size"
        ]

        heads = CONFIG[
            "num_heads"
        ]

        if size % heads != 0:

            raise ValueError(
                "embedding_size must be divisible by num_heads"
            )

        self.heads = heads

        self.head_size = (
            size // heads
        )

        self.query = nn.Linear(
            size,
            size,
            bias=False
        )

        self.key = nn.Linear(
            size,
            size,
            bias=False
        )

        self.value = nn.Linear(
            size,
            size,
            bias=False
        )

        self.output = nn.Linear(
            size,
            size
        )

        self.dropout = nn.Dropout(
            CONFIG["dropout"]
        )


    def forward(self, x):

        batch, time, channels = (
            x.shape
        )

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(
            batch,
            time,
            self.heads,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch,
            time,
            self.heads,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch,
            time,
            self.heads,
            self.head_size
        ).transpose(1, 2)


        attention = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_size
        )


        mask = torch.tril(
            torch.ones(
                time,
                time,
                device=x.device
            )
        )


        attention = attention.masked_fill(
            mask == 0,
            float("-inf")
        )


        attention = F.softmax(
            attention,
            dim=-1
        )


        attention = self.dropout(
            attention
        )


        result = (
            attention @ v
        )


        result = result.transpose(
            1,
            2
        ).contiguous()


        result = result.view(
            batch,
            time,
            channels
        )


        return self.output(
            result
        )


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(
    nn.Module
):

    def __init__(self):

        super().__init__()

        size = CONFIG[
            "embedding_size"
        ]


        self.network = nn.Sequential(

            nn.Linear(
                size,
                size * 4
            ),

            nn.GELU(),

            nn.Linear(
                size * 4,
                size
            ),

            nn.Dropout(
                CONFIG["dropout"]
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(
    nn.Module
):

    def __init__(self):

        super().__init__()

        size = CONFIG[
            "embedding_size"
        ]


        self.norm1 = nn.LayerNorm(
            size
        )

        self.attention = (
            CausalSelfAttention()
        )

        self.norm2 = nn.LayerNorm(
            size
        )

        self.feed_forward = (
            FeedForward()
        )


    def forward(self, x):

        x = (
            x +
            self.attention(
                self.norm1(x)
            )
        )

        x = (
            x +
            self.feed_forward(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# NOVA LANGUAGE MODEL
# ============================================================

class NovaLanguageModel(
    nn.Module
):

    def __init__(self):

        super().__init__()


        size = CONFIG[
            "embedding_size"
        ]


        self.token_embedding = (
            nn.Embedding(
                VOCAB_SIZE,
                size
            )
        )


        self.position_embedding = (
            nn.Embedding(
                CONFIG[
                    "context_length"
                ],
                size
            )
        )


        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(
                    CONFIG[
                        "num_layers"
                    ]
                )
            ]
        )


        self.final_norm = (
            nn.LayerNorm(size)
        )


        self.language_head = (
            nn.Linear(
                size,
                VOCAB_SIZE,
                bias=False
            )
        )


        self.language_head.weight = (
            self.token_embedding.weight
        )


    def forward(
        self,
        tokens,
        targets=None
    ):

        batch, time = (
            tokens.shape
        )


        positions = torch.arange(
            time,
            device=tokens.device
        )


        x = (
            self.token_embedding(
                tokens
            )
            +
            self.position_embedding(
                positions
            )
        )


        x = self.blocks(x)

        x = self.final_norm(x)

        logits = (
            self.language_head(x)
        )


        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    VOCAB_SIZE
                ),

                targets.reshape(-1)
            )


        return logits, loss


    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40
    ):

        self.eval()


        for _ in range(
            max_new_tokens
        ):

            context = tokens[
                :,
                -CONFIG[
                    "context_length"
                ]:
            ]


            logits, _ = self(
                context
            )


            logits = (
                logits[:, -1, :]
                /
                max(
                    temperature,
                    0.01
                )
            )


            if top_k:

                values, _ = (
                    torch.topk(
                        logits,
                        min(
                            top_k,
                            logits.size(-1)
                        )
                    )
                )


                logits[
                    logits <
                    values[:, [-1]]
                ] = float(
                    "-inf"
                )


            probabilities = (
                F.softmax(
                    logits,
                    dim=-1
                )
            )


            next_token = (
                torch.multinomial(
                    probabilities,
                    num_samples=1
                )
            )


            tokens = torch.cat(
                (
                    tokens,
                    next_token
                ),
                dim=1
            )


        return tokens


# ============================================================
# TRAINING BATCHES
# ============================================================

def get_batch():

    context_length = min(
        CONFIG[
            "context_length"
        ],
        len(
            encoded_training_data
        ) - 2
    )


    maximum_start = (
        len(
            encoded_training_data
        )
        -
        context_length
        -
        1
    )


    starts = torch.randint(
        0,
        maximum_start + 1,
        (
            CONFIG[
                "batch_size"
            ],
        )
    )


    x = torch.stack(
        [
            encoded_training_data[
                start:
                start +
                context_length
            ]

            for start in starts
        ]
    )


    y = torch.stack(
        [
            encoded_training_data[
                start + 1:
                start +
                context_length +
                1
            ]

            for start in starts
        ]
    )


    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# CREATE NOVA
# ============================================================

model = (
    NovaLanguageModel()
    .to(DEVICE)
)


parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


print(
    "Nova parameters:",
    f"{parameter_count:,}"
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG[
        "learning_rate"
    ],
    weight_decay=0.1
)


# ============================================================
# TRAIN
# ============================================================

print()
print(
    "Training Nova..."
)
print()


model.train()


for step in range(
    1,
    CONFIG[
        "training_steps"
    ] + 1
):

    x, y = get_batch()


    _, loss = model(
        x,
        y
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )


    optimizer.step()


    if (
        step == 1
        or
        step % 100 == 0
    ):

        print(
            f"Step {step:5d} | "
            f"Loss {loss.item():.4f}"
        )


# ============================================================
# SAVE NOVA
# ============================================================

BASE_DIRECTORY = (
    os.path.dirname(__file__)
)


MODEL_FILE = os.path.join(
    BASE_DIRECTORY,
    "nova-language-model.pt"
)


TOKENIZER_FILE = os.path.join(
    BASE_DIRECTORY,
    "nova-tokenizer.json"
)


CONFIG_FILE = os.path.join(
    BASE_DIRECTORY,
    "nova-language-config.json"
)


torch.save(
    model.state_dict(),
    MODEL_FILE
)


with open(
    TOKENIZER_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "stoi":stoi,
            "itos":{
                str(key):value
                for key,value
                in itos.items()
            }
        },
        file,
        indent=2,
        ensure_ascii=False
    )


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        CONFIG,
        file,
        indent=2
    )


print()
print(
    "Nova model saved:"
)
print(
    MODEL_FILE
)


# ============================================================
# QUICK GENERATION TEST
# ============================================================

prompt = (
    "<user>Hello</user>"
    "<assistant>"
)


prompt_tokens = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=DEVICE
)


generated = model.generate(
    prompt_tokens,
    max_new_tokens=50,
    temperature=0.8
)


print()
print(
    "Generation test:"
)
print()


print(
    decode(
        generated[0].tolist()
    )
)


print()
print(
    "Nova AI training complete."
)
