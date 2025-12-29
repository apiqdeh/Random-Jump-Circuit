#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

get_ipython().run_line_magic('matplotlib', 'inline')


# # Approximate Methods

# ## Random Jump Circuit

# In[ ]:


def geometric_weights(n: int):
    """
    Weights proportional to 1/2, 1/4, 1/8, ...
    """
    w = np.array([1 / (2**(k+1)) for k in range(n)]) # As said in the paper, here we introduce classical randomness
    return w / w.sum() # normalization of the weights that we'll later be used with a "rng.choice" function


# In[ ]:


def random_jump(num_pos_qubits: int = 4, num_steps: int = 10, seed: int = 0):
    """
    Random Jump circuit:
    - One coin qubit
    - Rx(pi/2) on the coin
    - For each step: H(coin), then CX(coin -> chosen position qubit)
    """
    rng = np.random.default_rng(seed) # randomness is always going to be the same for each 'seed' value
    w = geometric_weights(num_pos_qubits) 

    coin = num_pos_qubits  # put coin at the end 
    qc = QuantumCircuit(num_pos_qubits + 1, num_pos_qubits + 1)
    qc.rx(np.pi/2, coin) # prepare coin

    # Steps
    for _ in range(num_steps):
        qc.h(coin)
        j = int(rng.choice(num_pos_qubits, p=w))  # choose which position bit to "jump"
        qc.cx(coin, j)

    # Measure all (coin + position)
    qc.measure(range(num_pos_qubits + 1), range(num_pos_qubits + 1))
    return qc


# In[ ]:


# Example
qc_rj = random_jump(num_pos_qubits=4, num_steps=3, seed=7)
qc_rj.draw(output="mpl", fold=-1, style="iqp")
plt.show()


# ## Measurements

# In[ ]:


def counts_to_mean_position(counts: dict, num_pos_qubits: int) -> float:
    """
    Compute the mean (expected) position from Qiskit measurement counts.
    The input 'counts' contains bitstrings corresponding to (coin + position).

    """
    total_shots = sum(counts.values())
    weighted_sum = 0.0

    for bitstring, occurrences in counts.items():

        position_bits = bitstring[-num_pos_qubits:] # Extract only the position bits (ignore the coin bit)
        position_value = int(position_bits, 2) # Convert binary string to integer position
        weighted_sum += position_value * occurrences  # Add contribution weighted by how often it occurred

    return weighted_sum / total_shots # mean (expected) position


# ### Direct

# In[ ]:


def experiment_random_jump(
    num_pos_qubits: int = 8, # number of qubits
    num_steps: int = 9, # gate "layers"
    circuits: int = 30, # 30 different circuits
    shots: int = 30, # each circuit is runned 30 times
    base_seed: int = 123
):
    """
    Run multiple random instances of the Random Jump circuit and compute
    the mean measured position for each instance
    """
    sim = AerSimulator(seed_simulator=base_seed)
    mean_positions = []

    for k in range(circuits):
        seed = base_seed + k
        qc = random_jump(
            num_pos_qubits=num_pos_qubits,
            num_steps=num_steps,
            seed=seed
        )

        counts = sim.run(qc, shots=shots).result().get_counts(qc)
        mean_pos = counts_to_mean_position(counts, num_pos_qubits)
        mean_positions.append(mean_pos)

    return mean_positions


# In[ ]:


means = experiment_random_jump(
    num_pos_qubits=8,
    num_steps=9,
    circuits=30,
    shots=30
)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(means) + 1), means, marker='o')
plt.xlabel("Random circuit index")
plt.ylabel("Mean measured position")
plt.title("Random Jump Circuit (8 qubits, 9 steps)")
plt.grid(True)
plt.show()


# ### With post-processing

# Same as before, only this time we are using the substraction of the up and down positions

# In[ ]:


def experiment_random_jump_2(num_pos_qubits: int = 8, num_steps: int = 9, circuits: int = 30, shots: int = 30, base_seed: int = 123, 
seed_offset_down: int = 10_000):
    """
    Build a two-way walk from Random Jump circuits using classical post-processing:
        mean_position_two_way = mean_position_up - mean_position_down
    """
    #sim = AerSimulator(seed_simulator=base_seed)

    up_means = []
    down_means = []
    two_way_means = []

    for k in range(circuits):
        seed_up = base_seed + k
        seed_down = base_seed + seed_offset_down + k  # we use two different seeds for each circuit in order to avoid correlation

        sim_up = AerSimulator(seed_simulator=seed_up)     
        sim_down = AerSimulator(seed_simulator=seed_down)

        qc_up = random_jump(num_pos_qubits=num_pos_qubits, num_steps=num_steps, seed=seed_up) #"up" circuit with seed = base_seed + k
        qc_down = random_jump(num_pos_qubits=num_pos_qubits, num_steps=num_steps, seed=seed_down) #"down" circuit with seed = base_seed + seed_offset_down + k

        counts_up = sim_up.run(qc_up, shots=shots).result().get_counts(qc_up)
        counts_down = sim_down.run(qc_down, shots=shots).result().get_counts(qc_down)

        mu_up = counts_to_mean_position(counts_up, num_pos_qubits)
        mu_down = counts_to_mean_position(counts_down, num_pos_qubits)

        up_means.append(mu_up) # list of mean positions from the up circuits
        down_means.append(mu_down) # list of mean positions from the down circuits
        two_way_means.append(mu_up - mu_down) # list of length 'circuits' with signed mean positions

    return two_way_means, up_means, down_means


# In[ ]:


# Run
two_way_means, up_means, down_means = experiment_random_jump_2(num_pos_qubits=8, num_steps=9, circuits=30, shots=30, base_seed=123)


# In[ ]:


# Plot (signed mean positions)
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(two_way_means) + 1), two_way_means, marker="o")
plt.axhline(0, linewidth=1)
plt.xlabel("Random circuit index")
plt.ylabel("Mean position (up − down)")
plt.title("Two-way Random Jump (8 qubits, 9 steps) via classical post-processing")
plt.grid(True)
plt.show()

