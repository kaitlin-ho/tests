import numpy as np
import random
import requests

# for OpenAI stuff
from openai import OpenAI

# for PrefLib
import preflibtools as pt

### !!! Change this !!! ###
AUTHKEY = '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'
######### !!! #############
MODEL = "gpt-3.5-turbo-0125"


client = OpenAI(
    # This is the default and can be omitted
    api_key=AUTHKEY,
)

"""
Configuration stuff
"""

N = 2 # number of agents

M = 3 # number of items

num_instances = 1 # number of instances

# Define different pools of agent names
agentNamePools = {
    'default': ['Alice', 'Bob', 'Carol']
}

# Define different pools of item names
itemNamePools = {
    'fruits': ['apple', 'banana', 'orange'],
    'vegetables': ['asparagus', 'broccoli', 'potato'],
    'gems': ['diamond', 'emerald', 'ruby'],
    'houses': ['apartment', 'bungalow', 'villa'],
    'cars': ['buick', 'cadillac', 'ford']
}

# Define how many names should be sampled from each pool
agentPoolSampling = {
    'default': 2
}

for k in agentPoolSampling.keys():
    assert(k in agentNamePools)
    assert(agentPoolSampling[k] <= len(agentNamePools[k]))
assert(np.sum([agentPoolSampling[k] for k in agentPoolSampling.keys()]) == N)

itemPoolSampling = {
    'fruits': 2,
    'vegetables': 1
}

for k in itemPoolSampling.keys():
    assert(k in itemNamePools)
    assert(itemPoolSampling[k] <= len(itemNamePools[k]))
assert(np.sum([itemPoolSampling[k] for k in itemPoolSampling.keys()]) == M)

print("good config")

"""
Set agent and item names
TODO: Put it in a loop to generate a large number of instances
"""

agentNames = []
for k in agentPoolSampling.keys():
    agentNames += random.sample(agentNamePools[k], agentPoolSampling[k])

itemNames = []
for k in itemPoolSampling.keys():
    itemNames += random.sample(itemNamePools[k], itemPoolSampling[k])

"""
Set agents' valuations for items
TODO: Put it in a loop to generate a large number of instances
"""

def generate_valuations_IC(N, M, num_instances=1, low=1, high=10):
    """
    Generate instances where valuations are integers drawn from the uniform distribution over [low, high]
    Input:
        N: integer, number of agents
        M: integer, number of items
        num_instances: integer, number of instances to generate
        low: integer, lowest possible value of an item
        high: integer, highest possible value of an item
    Output:
        instances: a list of instances
                   each instances is a 2D N x M numpy array
                   whose i,j-th entry is the value that agent i has for item j
    """
    assert(low <= high)
    assert(num_instances > 0)
    instances = []
    for k in range(num_instances):
        val = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                val[i, j] = random.randint(low, high+1)
        instances.append(val)
    return instances

instances = generate_valuations_IC(N, M)

"""
Generate allocations
TODO: More sophisticated ways to generate allocations
"""

def generate_allocation_uniform(N, M):
    """
    Input:
        N: integer, number of agents
        M: integer, number of items
    Output:
        A: a 2D N x M numpy array
           the i, j-th entry is either 1 or 0 indicating whether agent i receives item j
    """
    A = np.zeros((N, M))
    for j in range(M):
        i = random.randint(0, N-1)
        A[i, j] = 1
    return A