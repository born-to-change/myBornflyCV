import mmcv

data = mmcv.load('../scripts/batch_rename.sh')

with open('test.json', file_format = 'json'):
    mmcv.dump(data, 'out.pkl')