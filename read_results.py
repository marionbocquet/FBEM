
import pickle

with open("FEM_Simulations.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
