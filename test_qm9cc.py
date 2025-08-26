import gzip
import json
# Import and print the pkl file in ./dataset_src/CC_test.json called test.pkl

with gzip.open('.dataset_src/den_graph_data_7_train.json.gz', 'rb') as f:
    data = json.load(f)
print(data[0])



