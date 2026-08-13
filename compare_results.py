import json
from pathlib import Path

mp = json.loads(Path('results/eval_mediapipe.json').read_text())
ft = json.loads(Path('results/eval_finetuned.json').read_text())

print()
print('=' * 62)
print('  BASELINE COMPARISON: MediaPipe vs Physics-Informed ViT')
print('=' * 62)
fmt = '{:<24s}  {:>10s}  {:>10s}  {:>10s}'
print(fmt.format('Metric', 'MediaPipe', 'Finetuned', 'Reduction'))
print('-' * 62)

def pct(a, b):
    return f'{(a - b) / a * 100:.1f}%'

mp_m = mp['overall_mpjpe_mm']
ft_m = ft['overall_mpjpe_mm']
print(fmt.format('Overall MPJPE (mm)', f'{mp_m:.2f}', f'{ft_m:.2f}', pct(mp_m, ft_m)))

labels = {'x': 'X-axis  (Width)', 'y': 'Y-axis  (Height)', 'z': 'Z-axis  (Depth) ***'}
for ax in ['x', 'y', 'z']:
    a = mp['axis_errors_mm'][ax]
    b = ft['axis_errors_mm'][ax]
    print(fmt.format(labels[ax], f'{a:.2f}', f'{b:.2f}', pct(a, b)))

print('-' * 62)
ft_z  = ft['axis_errors_mm']['z']
ft_xy = (ft['axis_errors_mm']['x'] + ft['axis_errors_mm']['y']) / 2
ratio = ft_xy / ft_z
print()
print(f'  Finetuned model Z/XY ratio  : {ft_z:.2f} mm Z  vs  {ft_xy:.2f} mm XY avg')
print(f'  Z error is {ratio:.2f}x LOWER than lateral errors')
print()
print('  THESIS CLAIM (slide-ready):')
print(f'  The Physics-Informed ViT reduces Z-axis depth hallucination')
mp_z = mp['axis_errors_mm']['z']
print(f'  from {mp_z:.2f} mm (MediaPipe) to {ft_z:.2f} mm (ours)')
print(f'  -- a {pct(mp_z, ft_z)} reduction -- using only a standard RGB camera.')
print('=' * 62)
