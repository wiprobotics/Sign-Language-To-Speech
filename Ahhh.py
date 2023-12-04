import pickle

# Open a pickle file and then print its contents

with open('ModelCreation/dataset/A-data.pickle', 'rb') as f:
    data = pickle.load(f)

print(data)