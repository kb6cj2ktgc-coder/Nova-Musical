"""
NOVA AI v0.1
Nova Musical

First trainable neural network for Nova Musical.

This model is created and trained by Nova Musical.
It does not download or use pretrained weights from
OpenAI, Google, Meta, Qwen, Gemma, or another LLM.
"""

import json
import math
import random


# ---------------------------------
# TRAINING DATA
# ---------------------------------

# INPUT:
# pitch error, accuracy, consistency, attempts
#
# OUTPUT:
# coaching action

training_data = [

    [-45, 0.40, 0.30, 1, "tune_up"],
    [-30, 0.50, 0.40, 2, "tune_up"],
    [-20, 0.65, 0.55, 3, "tune_up"],
    [-12, 0.75, 0.65, 3, "tune_up"],

    [-4, 0.80, 0.75, 2, "correct"],
    [-2, 0.90, 0.85, 3, "correct"],
    [0, 0.95, 0.90, 4, "correct"],
    [3, 0.88, 0.82, 3, "correct"],
    [5, 0.85, 0.80, 2, "correct"],

    [12, 0.75, 0.65, 3, "tune_down"],
    [20, 0.65, 0.55, 3, "tune_down"],
    [30, 0.50, 0.40, 2, "tune_down"],
    [45, 0.40, 0.30, 1, "tune_down"],

    [-6, 0.35, 0.30, 5, "repeat"],
    [7, 0.40, 0.35, 5, "repeat"],
    [3, 0.45, 0.40, 6, "repeat"],

    [1, 0.95, 0.95, 6, "advance"],
    [-1, 0.96, 0.94, 7, "advance"],
    [2, 0.98, 0.96, 8, "advance"]

]


actions = [
    "tune_up",
    "correct",
    "tune_down",
    "repeat",
    "advance"
]


# ---------------------------------
# SIMPLE NEURAL NETWORK
# ---------------------------------

INPUTS = 4
HIDDEN = 8
OUTPUTS = len(actions)


def random_weight():
    return random.uniform(-0.5, 0.5)


W1 = [
    [random_weight() for _ in range(HIDDEN)]
    for _ in range(INPUTS)
]

B1 = [
    0.0 for _ in range(HIDDEN)
]

W2 = [
    [random_weight() for _ in range(OUTPUTS)]
    for _ in range(HIDDEN)
]

B2 = [
    0.0 for _ in range(OUTPUTS)
]


def relu(x):
    return max(0.0, x)


def softmax(values):

    maximum = max(values)

    exps = [
        math.exp(v - maximum)
        for v in values
    ]

    total = sum(exps)

    return [
        value / total
        for value in exps
    ]


def normalize(row):

    cents = row[0] / 50
    accuracy = row[1]
    consistency = row[2]
    attempts = row[3] / 10

    return [
        cents,
        accuracy,
        consistency,
        attempts
    ]


def forward(inputs):

    hidden_raw = []

    for h in range(HIDDEN):

        value = B1[h]

        for i in range(INPUTS):
            value += (
                inputs[i] *
                W1[i][h]
            )

        hidden_raw.append(value)


    hidden = [
        relu(v)
        for v in hidden_raw
    ]


    output_raw = []

    for o in range(OUTPUTS):

        value = B2[o]

        for h in range(HIDDEN):
            value += (
                hidden[h] *
                W2[h][o]
            )

        output_raw.append(value)


    probabilities = softmax(
        output_raw
    )

    return (
        hidden_raw,
        hidden,
        probabilities
    )


# ---------------------------------
# TRAINING
# ---------------------------------

learning_rate = 0.03
epochs = 8000


for epoch in range(epochs):

    random.shuffle(training_data)

    total_loss = 0


    for row in training_data:

        x = normalize(row)

        target_name = row[4]

        target_index = (
            actions.index(target_name)
        )


        (
            hidden_raw,
            hidden,
            probabilities
        ) = forward(x)


        probability = max(
            probabilities[target_index],
            1e-12
        )

        total_loss += (
            -math.log(probability)
        )


        # Softmax + cross entropy gradient

        output_gradient = (
            probabilities.copy()
        )

        output_gradient[target_index] -= 1


        old_W2 = [
            row.copy()
            for row in W2
        ]


        # Update second layer

        for h in range(HIDDEN):

            for o in range(OUTPUTS):

                W2[h][o] -= (
                    learning_rate *
                    hidden[h] *
                    output_gradient[o]
                )


        for o in range(OUTPUTS):

            B2[o] -= (
                learning_rate *
                output_gradient[o]
            )


        # Hidden gradient

        hidden_gradient = [
            0.0
            for _ in range(HIDDEN)
        ]


        for h in range(HIDDEN):

            gradient = 0.0

            for o in range(OUTPUTS):

                gradient += (
                    old_W2[h][o] *
                    output_gradient[o]
                )


            if hidden_raw[h] <= 0:
                gradient = 0.0


            hidden_gradient[h] = gradient


        # Update first layer

        for i in range(INPUTS):

            for h in range(HIDDEN):

                W1[i][h] -= (
                    learning_rate *
                    x[i] *
                    hidden_gradient[h]
                )


        for h in range(HIDDEN):

            B1[h] -= (
                learning_rate *
                hidden_gradient[h]
            )


    if epoch % 1000 == 0:

        print(
            "Epoch:",
            epoch,
            "Loss:",
            round(total_loss, 4)
        )


# ---------------------------------
# TEST NOVA
# ---------------------------------

def predict(
    cents,
    accuracy,
    consistency,
    attempts
):

    x = normalize([
        cents,
        accuracy,
        consistency,
        attempts
    ])


    _, _, probabilities = (
        forward(x)
    )


    winner = max(
        range(len(probabilities)),
        key=lambda i:
            probabilities[i]
    )


    return (
        actions[winner],
        probabilities[winner]
    )


tests = [

    [-35, .60, .50, 2],

    [0, .90, .90, 3],

    [32, .60, .50, 2],

    [2, .98, .96, 8]

]


print("\nNOVA AI TEST\n")


for test in tests:

    decision, confidence = (
        predict(*test)
    )

    print(
        test,
        "=>",
        decision,
        round(confidence, 3)
    )


# ---------------------------------
# SAVE NOVA'S LEARNED MODEL
# ---------------------------------

model = {

    "name":
        "Nova AI",

    "version":
        "0.1",

    "architecture": {

        "inputs":
            INPUTS,

        "hidden":
            HIDDEN,

        "outputs":
            OUTPUTS

    },

    "actions":
        actions,

    "weights": {

        "W1":
            W1,

        "B1":
            B1,

        "W2":
            W2,

        "B2":
            B2

    }

}


with open(
    "nova-model.json",
    "w"
) as file:

    json.dump(
        model,
        file
    )


print(
    "\nNova AI training complete."
)

print(
    "Saved as nova-model.json"
)
