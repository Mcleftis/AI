import numpy as np
from hmmlearn import hmm

observations=np.array([[0],[0],[1],[0],[1],[1],[0],[0],[1],[0]])

model=hmm.MultinomialHMM(n_components=2, n_iter=100)
model.fit(observations)

hidden_states=model.predict(observations)
print("Observations (0=ilios, 1-=vroxi):", observvations.flatten())
print("Hidden states (0/1):", hidden_states)