import os
import random

data_dir = '/data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames'
seed = 8675309
train_size = .75
test_size = .20
val_size = .05

os.makedirs(data_dir + "/train", exist_ok=True)
os.makedirs(data_dir + "/val", exist_ok=True)
os.makedirs(data_dir + "/test", exist_ok=True)

file_names = os.listdir(data_dir)

video_ids = {}
for name in file_names:
    if name.endswith('npz'):
        video_id = name[:12]
        if not video_id in video_ids:
            video_ids[video_id] = [name]
        else:
            video_ids[video_id].append(name)

vids = list(video_ids.keys())
random.seed(seed)
random.shuffle(vids)

train = vids[:int(train_size*len(vids))]
val = vids[int(train_size*len(vids)):int(train_size*len(vids)+val_size*len(vids))]
test = vids[int(train_size*len(vids)+val_size*len(vids)):]

train_files = []
for vid in train:
    for npz in video_ids[vid]:
        train_files.append(npz)

val_files = []
for vid in val:
    for npz in video_ids[vid]:
        val_files.append(npz)

test_files = []
for vid in test:
    for npz in video_ids[vid]:
        test_files.append(npz)

print("These should be equal -> ", len(video_ids), len(train) + len(val) + len(test))
assert(len(video_ids) == len(train) + len(val) + len(test))

for npz_file in train_files:
    os.rename(os.path.join(data_dir, npz_file), os.path.join(data_dir, "train", npz_file))

for npz_file in val_files:
    os.rename(os.path.join(data_dir, npz_file), os.path.join(data_dir, "val", npz_file))

for npz_file in test_files:
    os.rename(os.path.join(data_dir, npz_file), os.path.join(data_dir, "test", npz_file))

with open(os.path.join(data_dir, 'train.csv'), 'w') as f:
    for npz_file in train_files:
        f.write(npz_file + "\n")

with open(os.path.join(data_dir, 'val.csv'), 'w') as f:
    for npz_file in val_files:
        f.write(npz_file + "\n")

with open(os.path.join(data_dir, 'test.csv'), 'w') as f:
    for npz_file in test_files:
        f.write(npz_file + "\n")
print(test_files)
