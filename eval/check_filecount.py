from pathlib import Path

gt_path = Path('../logs/test/PCL/gt')
gen_path = Path('../logs/test/PCL/gen')

gt_files = list(gt_path.glob('*.npz'))
gen_files = list(gen_path.glob('*.npz'))

for fn in gen_files:
    if fn.name not in [x.name for x in gt_files]:
        print(f'\033[31m[Error] {fn.name} have no ground truth\033[0m')

for fn in gt_files:
    if fn.name not in [x.name for x in gen_files]:
        print(f'[Warning] {fn.name} have no generated data')

print(f'gt files: {len(gt_files)}, gen files: {len(gen_files)}')

if len(gt_files) != len(gen_files):
    print('do you want to remove redundant ground truth? [Y/n]')
    if input().lower() == 'y':
        count = 0
        for fn in gt_files:
            if fn.name not in [x.name for x in gen_files]:
                fn.unlink()
                count += 1
        print(f'deleted {count} files')