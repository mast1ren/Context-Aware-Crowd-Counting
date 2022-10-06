import json
import os


def get_jpg_path(path, set):
    jpg_path = []
    
    for root, dirs, files in os.walk(os.path.join(path, set)):
        c = 0
        tmp_path = []
        for file in files:
            if file.endswith('.jpg'):
                tmp_path.append(os.path.join(root, file))
        tmp_path.sort()
        for path in tmp_path:
            if c % 5 == 0:
                jpg_path.append(path.replace('\\', '/'))
            c += 1

    print(len(jpg_path))
    
    jpg_path_json = json.dumps(jpg_path)
    with open(set + '.json', 'w') as f:
        f.write(jpg_path_json)
    f.close()

if __name__ == '__main__':
    for var in ['train', 'test', 'val']:
        get_jpg_path('../../ds/dronebird', var)