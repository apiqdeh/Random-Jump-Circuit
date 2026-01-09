#!/usr/bin/env python
# coding: utf-8

# # **'Approximate Quantum Counting for financial modeling'**
# 
# - **Author:** [Roque Mula Vela, Nuria Saniger Pulpillo, Pablo Sancho Villach, Álvaro Piqueras de Haro]
# 
# - **Date:** January 9, 2026
# 
# - **Subject:** Quantum Computing and Programming
# 
# - **Institution:** [Universitat Politècnica de Valencia]
# 

# #**Setup Environment**

# In[ ]:


# ALWAYS RUN THIS CELL AFTER YOU LOAD (OR RELOAD) THE NOTEBOOK

# Generic cell for correct operation of QCA materials in Google Colab (jcperez@disca.upv.es):
get_ipython().system('pip -qqq install qiskit[visualization]')
get_ipython().system('pip -qqq install qiskit-aer')
import qiskit
get_ipython().run_line_magic('matplotlib', 'inline')
qiskit.__version__

# Not always necessary (jcperez@disca.upv.es):
get_ipython().system('pip -qqq install git+https://github.com/qiskit-community/qiskit-textbook.git#subdirectory=qiskit-textbook-src')

# To fix a bug/version incompatibility in that file (jcperez@disca.upv.es):
# !sed -ie 's/denominator >/denominator() >/g' /usr/local/lib/python3.10/dist-packages/qiskit/visualization/array.py

# To set graphical circuit drawing by default in qiskit (jcperez@disca.upv.es):
get_ipython().system('mkdir ${HOME}/.qiskit 2>/dev/null')
get_ipython().system('printf "[default]\\ncircuit_drawer = mpl\\ncircuit_mpl_style = iqp\\n" > ${HOME}/.qiskit/settings.conf')


# In[ ]:


from qiskit import *
from qiskit.visualization import plot_distribution
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


pip install yfinance


# In[ ]:


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# download, 5 years of daily data of Dow Jones (^DJI)
data = yf.download("^DJI", period="5y", interval="1d", auto_adjust=True)

close = data["Close"].values

# daily retunrs
returns_real = (close[1:] - close[:-1]) / close[:-1]

mu_real = returns_real.mean()
sigma_real = returns_real.std()

print("Mean returns DJIA:", mu_real)
print("Sigma returns DJIA:", sigma_real)


# 
# 
# ## **1) Arc Counter Circuit**

# In[ ]:


#We first define a function that creates our quantum circuit, with a specific number of steps and qubits
def arc_counter_circuit(steps,n_qubits):
  qc = QuantumCircuit(n_qubits)
  for j in range (steps):
    n=2
    for i in range (n_qubits):
      qc.rx(pi/n,i)
      n= 2*n
  qc.measure_all()

  return qc


# In[ ]:


#This is just to visualize our circuit and verify that it is well designed
n_qubits=8
qc= arc_counter_circuit(1,4)
qc.draw()


# In[ ]:


def counts_to_positions(counts):
#With this function we convert bitstrings to integers, where q0 is the LSB because is the one that rotates faster
# Qiskit takes the bitstring with MSB on the left
    data = []
    for bitstring, c in counts.items():
        x = int(bitstring,2)
        data.extend([x] * c) #number of shots of x
    return data


# In[ ]:


#This function generates the plots of the counts vs position

def plot_steps(ax, steps_list, n_qubits=8, shots=1000, xlim=None, mark_peaks = True):
    backend = AerSimulator()
    bins = range(0, 2**n_qubits + 1)

    for steps in steps_list:
        qc = arc_counter_circuit(steps, n_qubits)
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts() #This extracts a dictionary with the bitstrings and in the number of shots that they been obtained
        positions = counts_to_positions(counts)
        hist, edges = np.histogram(positions, bins=bins)  # counts por bin
        centers = edges[:-1] + np.diff(edges)/2           # centros de bin

        ax.plot(centers, hist, linewidth=1.2, label=str(steps))

        #This is for drawing points in the peaks
        if mark_peaks:
            y = hist
            peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1  # local maximums
            peaks = np.unique(np.r_[peaks, np.argmax(y)])  #this include the absolute maximum
            ax.scatter(centers[peaks], y[peaks], s=20)

    #ax.set_xlim(0, 2**n_qubits - 1)
    ax.set_xlim(0,70)
    ax.set_xlabel("Position (integer at the register)")
    ax.set_ylabel("Counts (shots)")
    ax.legend(title="Steps")
    ax.grid(True, which='both', linestyle='--', alpha=0.35)


# In[ ]:


#Generate figure with two subplots
n_qubits = 8
shots = 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

plot_steps(ax1, steps_list=[1,5,10,20,25], n_qubits=n_qubits, shots=shots, xlim=(0, 2**n_qubits - 1))

#plot_steps(ax1, steps_list=[1,2,3,4,5], n_qubits=n_qubits, shots=shots, xlim=(0, 2**n_qubits - 1))
plot_steps(ax2, steps_list=[20,21,22,23,24,25], n_qubits=n_qubits, shots=shots, xlim=(0, 2**n_qubits - 1))

plt.show()


# In[ ]:


from qiskit import transpile

def sample_arc_counter_positions(num_steps, n_qubits=8, shots=1000):
    backend = AerSimulator()
    qc = arc_counter_circuit(num_steps, n_qubits)
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    positions = counts_to_positions(counts)   # << aquí usas tu función
    return np.array(positions)


# In[ ]:





# In[ ]:


# We assure 1D for real data
pos_signed = sample_arc_counter_positions(num_steps=20, n_qubits=8, shots=10000)
returns_real = np.ravel(returns_real)

# Sigma of the positions of the random Jump
sigma_pos = pos_signed.std()
print("Sigma Counter positions:", sigma_pos)

sigma_real = 0.009266736595833835  # tsigma from DJIA

k_rj = sigma_pos / sigma_real       # reescalate factor
returns_rj_rescaled = pos_signed / k_rj

mu_rj = returns_rj_rescaled.mean()
sigma_rj = returns_rj_rescaled.std()
print("Mean RJ reescalated:", mu_rj)
print("Sigma RJ reescalated:", sigma_rj)


# In[ ]:


returns_rj_rescaled = np.ravel(returns_rj_rescaled)

all_data = np.concatenate([returns_real, returns_rj_rescaled])
xmin, xmax = np.percentile(all_data, [0.5, 99.5])
bins = 50

plt.figure(figsize=(6,4))

plt.hist(returns_real,
         bins=bins,
         range=(xmin, xmax),
         density=True,
         alpha=0.6,
         color="tab:blue",
         label="Dow Jones ")

plt.hist(returns_rj_rescaled,
         bins=bins,
         range=(xmin, xmax),
         density=True,
         alpha=0.6,
         color="tab:purple",
         label="Escalated Arc counter ")

plt.xlabel("Daily returns")
plt.ylabel("Density")
plt.title("Returns: Dow Jones vs Arc Counter (20 steps)")
plt.legend()


from google.colab import files

plt.tight_layout()
plt.grid(True, alpha=0.3)

plt.savefig("dow_vs_arcwalk_two_way.png", dpi=300, bbox_inches="tight")
plt.show()

# Descarga al ordenador (irá a la carpeta Descargas del navegador)
files.download("dow_vs_arcwalk_two_way.png")



# # 2) **Random Jump Circuit**
# 

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


# ### With Post-Processing

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


# In[ ]:


# Aseguramos 1D para los datos reales
pos_signed = np.array(two_way_means)
returns_real = np.ravel(returns_real)

# Sigma de las posiciones del Random Jump
sigma_pos = pos_signed.std()
print("Sigma posiciones Random Jump:", sigma_pos)

sigma_real = 0.009266736595833835  # tsigma fromDJIA

k_rj = sigma_pos / sigma_real       # reescalate factor
returns_rj_rescaled = pos_signed / k_rj

mu_rj = returns_rj_rescaled.mean()
sigma_rj = returns_rj_rescaled.std()
print("Media RJ reescalado:", mu_rj)
print("Sigma RJ reescalado:", sigma_rj)


# In[ ]:


returns_rj_rescaled = np.ravel(returns_rj_rescaled)

all_data = np.concatenate([returns_real, returns_rj_rescaled])
xmin, xmax = np.percentile(all_data, [0.5, 99.5])
bins = 50

plt.figure(figsize=(6,4))

plt.hist(returns_real,
         bins=bins,
         range=(xmin, xmax),
         density=True,
         alpha=0.6,
         color="tab:blue",
         label="Dow Jones")

plt.hist(returns_rj_rescaled,
         bins=bins,
         range=(xmin, xmax),
         density=True,
         alpha=0.6,
         color="tab:green",
         label=" Escalated Random Jump")

plt.xlabel("Daily returns")
plt.ylabel("Density")
plt.title("Returns: Dow Jones vs Random Jump (9 steps)")
plt.legend()

from google.colab import files

plt.tight_layout()
plt.grid(True, alpha=0.3)

plt.savefig("dow_vs_arcwalk_two_way.png", dpi=300, bbox_inches="tight")
plt.show()

# Descarga al ordenador (irá a la carpeta Descargas del navegador)
files.download("dow_vs_arcwalk_two_way.png")



# **Arc walker**

# In[ ]:


from qiskit.visualization import circuit_drawer
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# parámetros
num_counter = 8
num_steps   = 10          # por ejemplo, 10 como en la figura
base_theta  = np.pi/2

# q0 = coin, q1..q8 = arc counter
q = QuantumRegister(1 + num_counter, 'q')
c = ClassicalRegister(num_counter, 'c')
qc = QuantumCircuit(q, c, name="arc_walk_counter")

coin    = 0
counter = list(range(1, 1 + num_counter))

# bloques: H en la coin + CRx en cada qubit del contador,
# con ángulos escalados 1, 1/2, 1/4, ...
for _ in range(num_steps):
    qc.h(q[coin])
    for idx, qi in enumerate(counter):
        angle = base_theta / (2**idx)
        qc.crx(angle, q[coin], q[qi])

# mediciones en el contador (igual que en la simulación)
for i, qi in enumerate(counter):
    qc.measure(q[qi], c[i])

# dibujar en formato mpl
fig = circuit_drawer(qc, output='mpl', scale=1.5)
fig


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# 1. Backends: ideal y and noisy

sim_ideal = AerSimulator()

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.001, 1), ['rx', 'ry', 'rz', 'h', 'x']
)
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 2), ['cx', 'crx']
)
sim_noisy = AerSimulator(noise_model=noise_model,
                         basis_gates=noise_model.basis_gates)


# 2. Arc Walk counter

def arc_walk_counter(num_counter_qubits=8, num_steps=20, base_theta=np.pi/2):
    N = num_counter_qubits
    q = QuantumRegister(1 + N, 'q')
    c = ClassicalRegister(N, 'c')
    qc = QuantumCircuit(q, c)

    coin = 0
    counter = [i for i in range(1, N+1)]

    for _ in range(num_steps):
        qc.h(coin)
        for idx, qi in enumerate(counter):
            angle = base_theta / (2**idx)   # θ/2, θ/4, ...
            qc.crx(angle, coin, qi)

    for i, qi in enumerate(counter):
        qc.measure(qi, c[i])

    return qc

def run_arc_walk(backend, num_counter_qubits, num_steps, base_theta, shots=1000):
    qc = arc_walk_counter(num_counter_qubits, num_steps, base_theta)
    t_qc = transpile(qc, backend)
    result = backend.run(t_qc, shots=shots).result()
    return result.get_counts()

def counts_to_positions(counts):
    positions = []
    for bitstr, cnt in counts.items():
        pos = int(bitstr, 2)
        positions.extend([pos] * cnt)
    return np.array(positions)

def two_way_arc_walk_counts(backend, num_counter_qubits, num_steps, base_theta, shots):
    # up
    counts_up = run_arc_walk(backend, num_counter_qubits, num_steps, base_theta, shots)
    pos_up = counts_to_positions(counts_up)
    # down
    counts_down = run_arc_walk(backend, num_counter_qubits, num_steps, -base_theta, shots)
    pos_down = counts_to_positions(counts_down)
    # concatenamos +p y -p
    return np.concatenate([pos_up, -pos_down])

# ================================
# 3. Parámetros comunes
# ================================
shots = 1000
num_counter = 8
base_theta = np.pi/2
steps_list = range(10, 16)


# In[ ]:


# ===== Figura 1: one-way (+) =====
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))

for s in steps_list:
    # ideal
    counts_id = run_arc_walk(sim_ideal, num_counter, s, base_theta, shots)
    pos_id = counts_to_positions(counts_id)
    vals_id, freqs_id = np.unique(pos_id, return_counts=True)
    axes1[0].plot(vals_id, freqs_id, marker='.', linestyle='-', label=f"{s} (ideal)")

    # noisy
    counts_no = run_arc_walk(sim_noisy, num_counter, s, base_theta, shots)
    pos_no = counts_to_positions(counts_no)
    vals_no, freqs_no = np.unique(pos_no, return_counts=True)
    axes1[1].plot(vals_no, freqs_no, marker='.', linestyle='-', label=f"{s} (noisy)")

axes1[0].set_xlim(0, 200)
axes1[1].set_xlim(0, 200)
axes1[1].set_ylim(0, 250)

axes1[0].set_title("Arc walk counter (only +, ideal)")
axes1[1].set_title("Arc walk counter (only +, noisy)")
for ax in axes1:
    ax.set_xlabel("Position")
    ax.grid(True)
axes1[0].set_ylabel("Frequency")
axes1[0].legend(title="Steps")
axes1[1].legend(title="Steps")
plt.tight_layout()
plt.show()

# ===== Figura 2: two-way =====
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

for s in steps_list:
    diffs_id = two_way_arc_walk_counts(sim_ideal, num_counter, s, base_theta, shots)
    vals_id, freqs_id = np.unique(diffs_id, return_counts=True)
    axes2[0].plot(vals_id, freqs_id, marker='.', linestyle='-', label=f"{s} (ideal)")

    diffs_no = two_way_arc_walk_counts(sim_noisy, num_counter, s, base_theta, shots)
    vals_no, freqs_no = np.unique(diffs_no, return_counts=True)
    axes2[1].plot(vals_no, freqs_no, marker='.', linestyle='-', label=f"{s} (noisy)")

axes2[0].set_xlim(-200, 200)
axes2[1].set_xlim(-200, 200)

axes2[1].set_ylim(0, 250)

axes2[0].set_title("Arc walk counter (two-way, ideal)")
axes2[1].set_title("Arc walk counter (two-way, noisy)")
for ax in axes2:
    ax.set_xlabel("Position")
    ax.grid(True)
axes2[0].set_ylabel("Frequency")
axes2[0].legend(title="Steps")
axes2[1].legend(title="Steps")
plt.tight_layout()
plt.show()


# In[ ]:


num_steps_tw = max(steps_list)

# diffs_noisy: posiciones firmadas (up − down) del Arc Walk con ruido
diffs_noisy = two_way_arc_walk_counts(sim_noisy, num_counter, num_steps_tw, base_theta, shots)

# Esto será nuestro pos_signed para el Arc Walk ruidoso
pos_signed = np.array(diffs_noisy, dtype=float)

# Aseguramos 1D para los datos reales
returns_real = np.ravel(returns_real)

# Sigma de las posiciones del Arc Walk two-way (noisy)
sigma_pos = pos_signed.std()
print("Sigma posiciones Arc Walk two-way (noisy):", sigma_pos)

# Sigma del DJIA (ya calculada antes)
sigma_real = 0.009266736595833835

# Factor de reescalado
k_aw = sigma_pos / sigma_real
returns_aw_rescaled = pos_signed / k_aw


# In[ ]:


mu_aw = returns_aw_rescaled.mean()
sigma_aw = returns_aw_rescaled.std()
#print("Media Arc Walk reescalado:", mu_aw)
#print("Sigma Arc Walk reescalado:", sigma_aw)

# ---------- Histograma comparando con DJIA ----------
returns_aw_rescaled = np.ravel(returns_aw_rescaled)

all_data = np.concatenate([returns_real, returns_aw_rescaled])
xmin, xmax = np.percentile(all_data, [0.5, 99.5])
bins = 50

plt.figure(figsize=(6,4))

plt.hist(returns_real,
         bins=bins,
         range=(xmin, xmax),
         density=True,
         alpha=0.6,
         color="tab:blue",
         label="Dow Jones")

plt.hist(returns_aw_rescaled,
         bins=bins,
         range=(xmin, xmax),
         density=True,
         alpha=0.6,
         color="tab:orange",
         label="Escalated noisy two-way Arc Walk")

plt.xlabel("Daily return")
plt.ylabel("Density")
plt.title(f"Returns: Dow Jones vs Arc Walk two-way (noisy, {num_steps_tw} steps)")


plt.legend(title="Series",
           fontsize=8,           # tamaño del texto
           title_fontsize=9,     # tamaño del título
           framealpha=0.7,       # cuadro semitransparente
           loc="upper left")     # posición


from google.colab import files

plt.tight_layout()
plt.grid(True, alpha=0.3)

plt.savefig("dow_vs_arcwalk_two_way.png", dpi=300, bbox_inches="tight")
plt.show()

# Descarga al ordenador (irá a la carpeta Descargas del navegador)
files.download("dow_vs_arcwalk_two_way.png")





# ##Quantum Counting Estandar (QPE + Iterador de Grover)
# 
# 

# In[ ]:


import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator

# 1. Function to generate the circuit
def build_standard_quantum_counting(num_state_qubits: int = 4, num_counting_qubits: int = 3) -> QuantumCircuit:
    counting_reg = QuantumRegister(num_counting_qubits, name='count')
    state_reg = QuantumRegister(num_state_qubits, name='state')
    c_reg = ClassicalRegister(num_counting_qubits, name='meas')

    qc = QuantumCircuit(counting_reg, state_reg, c_reg)

    # Inicialization
    qc.h(counting_reg)
    qc.h(state_reg)

    # Oracle (same as before)
    oracle = QuantumCircuit(num_state_qubits)
    oracle.h(num_state_qubits - 1)
    oracle.mcx(list(range(num_state_qubits - 1)), num_state_qubits - 1)
    oracle.h(num_state_qubits - 1)

    # Grover operator
    grover_op = GroverOperator(oracle)

    # QPE
    for j in range(num_counting_qubits):
        controlled_grover = grover_op.power(2**j).control()
        qc.append(controlled_grover, [counting_reg[j]] + list(state_reg))

    # Inverse QFT
    qc.append(QFT(num_counting_qubits, inverse=True), counting_reg)


    qc.measure(counting_reg, c_reg)

    return qc

# Analysis function
def calculate_solutions_from_counts(counts, num_state_qubits, num_counting_qubits):
    N = 2**num_state_qubits
    results = {}
    for outcome_string, count in counts.items():
        measured_int = int(outcome_string, 2)
        theta = (measured_int / (2**num_counting_qubits)) * 2 * math.pi
        M_estimated = N * (math.sin(theta / 2) ** 2)
        results[outcome_string] = {
            "measured_int": measured_int,
            "estimated_M": round(M_estimated),
            "shots": count
        }
    return results

# 3. CORRECTED EXECUTION
print("Generating circuit...")
#We use little amount of qubits for not taking so long
qc_standard = build_standard_quantum_counting(num_state_qubits=4, num_counting_qubits=4)

print("Iniciando simulador...")
sim = AerSimulator()

# This is the arrangemet
print("Transpiling circuit")
qc_transpiled = transpile(qc_standard, sim)

print("Executing simulation")
result = sim.run(qc_transpiled, shots=2048).result()
counts = result.get_counts()

# Analysis
analysis = calculate_solutions_from_counts(counts, num_state_qubits=4, num_counting_qubits=4)
most_frequent = max(counts, key=counts.get)

print(f"\nMost frequent result: {most_frequent}")
print(f"Estimated solutions (M): {analysis[most_frequent]['estimated_M']}")


# In[ ]:


import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

#We adjust the parameters for being executable but illustrative
NUM_STATE_QUBITS = 4    # Search space
NUM_COUNTING_QUBITS = 4 # Precision (t=4)
SHOTS = 2048

# Construction of the circuit
def build_standard_quantum_counting(n_state, n_count):
    """
    Construye el circuito de Quantum Counting usando QPE sobre el Operador de Grover.
    """
    # Registers
    counting_reg = QuantumRegister(n_count, name='count')
    state_reg = QuantumRegister(n_state, name='state')
    c_reg = ClassicalRegister(n_count, name='meas')

    qc = QuantumCircuit(counting_reg, state_reg, c_reg)

    # A. Inicialization
    qc.h(counting_reg)
    qc.h(state_reg)

    # B. Oracle
    oracle = QuantumCircuit(n_state)
    oracle.h(n_state - 1)
    oracle.mcx(list(range(n_state - 1)), n_state - 1)
    oracle.h(n_state - 1)

    # C. Grover operator
    grover_op = GroverOperator(oracle)

    # D. Controlled Quantum Phase Estimation (QPE)
    # Here is the compplexity
    for j in range(n_count):
        controlled_grover = grover_op.power(2**j).control()
        #We apply the control from the qubit j
        qc.append(controlled_grover, [counting_reg[j]] + list(state_reg))

    # E.QFT†
    qc.append(QFT(n_count, inverse=True), counting_reg)

    qc.measure(counting_reg, c_reg)

    return qc


print(f"1. Generating Standard Quantum Counting circuit (n={NUM_STATE_QUBITS}, t={NUM_COUNTING_QUBITS})...")
qc_standard = build_standard_quantum_counting(NUM_STATE_QUBITS, NUM_COUNTING_QUBITS)

print("2. Transpiling circuit")
sim = AerSimulator()
qc_transpiled = transpile(qc_standard, sim)

print(f"3. Executions ({SHOTS} shots)...")
result = sim.run(qc_transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Analysis of results
def analyze_results(counts, n_state, n_count):
    N = 2**n_state # Total de estados en el espacio de búsqueda
    data = {}

    # We search the ms¡ost frequent result
    most_frequent_bitstring = max(counts, key=counts.get)

    for bitstring, count in counts.items():
        measured_int = int(bitstring, 2)

        theta = (measured_int / (2**n_count)) * 2 * math.pi

        M_estimated = N * (math.sin(theta / 2) ** 2)

        data[bitstring] = round(M_estimated)

    return most_frequent_bitstring, data

best_bitstring, estimated_solutions = analyze_results(counts, NUM_STATE_QUBITS, NUM_COUNTING_QUBITS)



# A. Draw the circuit
# We use 'fold=-1' for not being cut, although it will be so long
print("\n--- Diagram of the circuit (Collapsed version of high level)")
display(qc_standard.draw('mpl', fold=-1))

# B. Plot
print("\n--- RESULTS: HISTOGRAM OF MEASURE PHASES")
print("X axis shows the binary phase measured. The peak indicates the correct value")
display(plot_histogram(counts))


print("\nFinal Interpretation")
print(f"Most frequent bitstring: {best_bitstring}")
print(f"Integer value measured (phase): {int(best_bitstring, 2)}")
print(f"Estimated number of solutions (M): {estimated_solutions[best_bitstring]}")
print(f"(The theoretical value is 1, because we are marking just one state)")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator

# Constructor of the circuit (baseline)
def build_standard_qc(num_state_qubits, num_counting_qubits):
    """
    Construye el circuito de conteo estándar (QPE + Grover).
    """
    # Registers
    counting_reg = QuantumRegister(num_counting_qubits, name='count')
    state_reg = QuantumRegister(num_state_qubits, name='state')
    c_reg = ClassicalRegister(num_counting_qubits, name='meas')

    qc = QuantumCircuit(counting_reg, state_reg, c_reg)

    # Inicialization
    qc.h(counting_reg)
    qc.h(state_reg)

    # Oracle: We mark the state |11...1> (M=1 Real solution)
    oracle = QuantumCircuit(num_state_qubits)
    oracle.h(num_state_qubits - 1)
    oracle.mcx(list(range(num_state_qubits - 1)), num_state_qubits - 1)
    oracle.h(num_state_qubits - 1)


    grover_op = GroverOperator(oracle)

    # QPE (Applying controlled powers: exponential complexity)
    for j in range(num_counting_qubits):
        controlled_grover = grover_op.power(2**j).control()
        qc.append(controlled_grover, [counting_reg[j]] + list(state_reg))


    qc.append(QFT(num_counting_qubits, inverse=True), counting_reg)


    qc.measure(counting_reg, c_reg)

    return qc

# Data processing
def get_m_distribution(counts, n_state, n_count):
    """
    Converts the bitstrings measured (phases) to number of estimated solutions (M).
    This normalizes x axis to compare different precisions.
    """
    N = 2**n_state
    # Range of possible solutions (0 to N)
    x_domain = np.arange(0, N + 1)
    y_values = np.zeros(len(x_domain))

    for bitstring, shots in counts.items():
        # 1. Binary phase -> integer
        measured_int = int(bitstring, 2)

        # 2. Integer -> Theta phase
        theta = (measured_int / (2**n_count)) * 2 * math.pi

        # 3. Theta phase -> M Estimation (Grover's formula)
        M_estimated = N * (math.sin(theta / 2) ** 2)
        M_round = round(M_estimated)

        # We acumulate shots in the nearest integer
        if 0 <= M_round <= N:
            y_values[M_round] += shots

    return x_domain, y_values

# Execution and plot
def run_comparison_plot(n_state=4, precision_list=[3, 4, 5, 6], shots=2048):
    sim = AerSimulator()


    fig, ax = plt.subplots(figsize=(10, 6))


    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    print(f"Iniciating simulation for N={2**n_state} states")

    last_qc = None # We store the last to draw

    for i, t in enumerate(precision_list):
        print(f"  -> Simulating with t={t} qubits of precision")


        qc = build_standard_qc(n_state, t)
        last_qc = qc


        qc_trans = transpile(qc, sim)
        result = sim.run(qc_trans, shots=shots).result()
        counts = result.get_counts()


        x, y = get_m_distribution(counts, n_state, t)


        color = colors[i % len(colors)]
        ax.plot(x, y, label=f'Precision $t={t}$',
                color=color, linewidth=1.5, marker='o', markersize=3, alpha=0.8)


    ax.set_title(f"Standard Quantum Counting (Baseline)\nTarget: 1 Solution in {2**n_state} states", fontsize=14)
    ax.set_xlabel("Estimated Number of Solutions ($M$)", fontsize=12)
    ax.set_ylabel("Counts (Shots)", fontsize=12)
    ax.legend(title="Counting Qubits", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)


    ax.set_xlim(-0.5, 6.5)

    plt.tight_layout()
    plt.show()

    return last_qc

#We simulate a search space of 4 qubits (16 elements)
# Precision (t) from 3 to 6 qubits.
circuit_final = run_comparison_plot(n_state=4, precision_list=[3, 4, 5, 6], shots=4096)

print("\nStandard Circuit")
print("Observe depth due to the Grover powers")
circuit_final.draw('mpl', fold=-1)


# In[ ]:


# Noise graph
from qiskit_aer.noise import NoiseModel, depolarizing_error


noise_model = NoiseModel()
error_gate = depolarizing_error(0.02, 1) # Error in gates of 1 qubit
error_cnot = depolarizing_error(0.05, 2) # Error in gates of 2 qubits (CNOT) highest

noise_model.add_all_qubit_quantum_error(error_gate, ["u1", "u2", "u3", "h"])
noise_model.add_all_qubit_quantum_error(error_cnot, ["cx"])


print("Simulando con ruido...")
sim_noise = AerSimulator(noise_model=noise_model)


qc = build_standard_qc(num_state_qubits=4, num_counting_qubits=4)
qc_trans = transpile(qc, sim_noise)

result_noise = sim_noise.run(qc_trans, shots=4096).result()
counts_noise = result_noise.get_counts()


x_noise, y_noise = get_m_distribution(counts_noise, 4, 4)

# 3. Plot comparison (Ideal vs Noise)
plt.figure(figsize=(10, 6))
plt.plot(x_noise, y_noise, label='Standard QC with Noise', color='red', marker='x')
plt.title("Catastrophic Failure of Standard Counting under Noise")
plt.xlabel("Estimated Solutions (M)")
plt.ylabel("Counts")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[ ]:


get_ipython().system('pip -qqq install qiskit[visualization]')
get_ipython().system('pip -qqq install qiskit-aer')

import numpy as np
import matplotlib.pyplot as plt
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import GroverOperator, QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# --- 1. BASELINE ---
def build_standard_qc(num_state_qubits, num_counting_qubits):
    """
    Builds the standard quantum counting circuit (QPE + Grover).
    """
    # Registers
    counting_reg = QuantumRegister(num_counting_qubits, name='count')
    state_reg = QuantumRegister(num_state_qubits, name='state')
    c_reg = ClassicalRegister(num_counting_qubits, name='meas')

    qc = QuantumCircuit(counting_reg, state_reg, c_reg)

    # Initialization
    qc.h(counting_reg)
    qc.h(state_reg)

    # Oracle
    oracle = QuantumCircuit(num_state_qubits)
    oracle.h(num_state_qubits - 1)
    oracle.mcx(list(range(num_state_qubits - 1)), num_state_qubits - 1)
    oracle.h(num_state_qubits - 1)

    # Grover Operator
    grover_op = GroverOperator(oracle)

    # QPE
    for j in range(num_counting_qubits):
        controlled_grover = grover_op.power(2**j).control()
        qc.append(controlled_grover, [counting_reg[j]] + list(state_reg))

    # Inverse QFT
    qc.append(QFT(num_counting_qubits, inverse=True), counting_reg)

    # Measurement
    qc.measure(counting_reg, c_reg)

    return qc

# Configuration
n_state = 4
n_count = 4  # We use t=4 as a trade-off
shots = 2048
correct_state_decimal = 1  # We know the correct answer is 1

# Range of error rates to test (from 0% to 5%)
error_rates = np.linspace(0.000, 0.05, 10)
success_probs = []

print("Simulating degradation curve...")

qc_base = build_standard_qc(n_state, n_count)  # Function defined above

for p in error_rates:
    noise_model = NoiseModel()
    error_gate = depolarizing_error(p, 1)
    error_cnot = depolarizing_error(p * 2, 2)  # Assume CNOT is twice as noisy
    noise_model.add_all_qubit_quantum_error(error_gate, ["u1", "u2", "u3", "h", "x"])
    noise_model.add_all_qubit_quantum_error(error_cnot, ["cx"])

    # Simulator
    sim = AerSimulator(noise_model=noise_model)
    qc_trans = transpile(qc_base, sim)
    counts = sim.run(qc_trans, shots=shots).result().get_counts()

    # Probability of success
    hits = 0
    for bitstring, count in counts.items():
        val = int(bitstring, 2)
        # Convert to estimated M
        theta = (val / (2**n_count)) * 2 * np.pi
        M_est = 2**n_state * (np.sin(theta / 2)**2)
        if round(M_est) == correct_state_decimal:
            hits += count

    prob = hits / shots
    success_probs.append(prob)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(error_rates * 100, success_probs, 'o-', color='purple', linewidth=2)
plt.title(f"Sensitivity Analysis: Standard Counting (t={n_count})")
plt.xlabel("Gate Error Rate (%)")
plt.ylabel("Probability of Success (Measuring M=1)")
plt.grid(True, linestyle='--')
plt.axhline(y=1.0, color='green', linestyle=':', label='Ideal')
plt.axhline(y=1/16, color='gray', linestyle=':', label='Random Guessing')
plt.legend()
plt.show()


# In[ ]:


Ms = []  # list with the estimated M values per shot
for bitstring, shots in counts.items():
    measured_int = int(bitstring, 2)
    theta = (measured_int / (2**NUM_COUNTING_QUBITS)) * 2 * math.pi
    M_estimated = 2**NUM_STATE_QUBITS * (math.sin(theta / 2)**2)
    M_rounded = round(M_estimated)
    Ms.extend([M_rounded] * shots)

Ms = np.array(Ms, dtype=float)

# Convert to centered "returns"
returns_qc_raw = Ms - 1.0   # if the theoretical M is 1

# Rescaling of sigma as before
sigma_real = returns_real.std()
sigma_qc = returns_qc_raw.std()
k_qc = sigma_qc / sigma_real
returns_qc_rescaled = returns_qc_raw / k_qc


# In[ ]:


all_data = np.concatenate([returns_real, returns_qc_rescaled])
xmin, xmax = np.percentile(all_data, [0.5, 99.5])
bins = 50

plt.figure(figsize=(6, 4))
plt.hist(returns_real, bins=bins, range=(xmin, xmax),
         density=True, alpha=0.6, color="tab:blue",
         label="Real Dow Jones")
plt.hist(returns_qc_rescaled, bins=bins, range=(xmin, xmax),
         density=True, alpha=0.6, color="tab:red",
         label="Rescaled Standard QC")
plt.xlabel("Daily return")
plt.ylabel("Density")
plt.title("Returns: Dow Jones vs Standard Quantum Counting")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:




