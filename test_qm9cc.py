import pickle
import json
# Import and print the pkl file in ./dataset_src/CC_test.json called test.pkl

with open('.dataset_src/CC_test.json/test.pkl', 'rb') as f:
    data = json.load(f)
print(data)



