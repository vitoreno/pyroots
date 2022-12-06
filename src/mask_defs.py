from matplotlib.patches import Patch

colors = {"red": [255, 0, 0], "black": [0, 0, 0], "green": [0, 255, 0], "super_green": [0, 150, 0], "yellow": [255, 255, 0], "orange": [255, 128, 0], "blue": [0, 0, 255]}
colors_float = {key:[v/255.0 for v in value] for (key,value) in colors.items()}

black_patch = Patch(color=colors_float["black"], label=r'$\sigma \leq$ 0.3')
yellow_patch = Patch(color=colors_float["yellow"], label=r'0.3 < $\sigma$ < 0.6')
orange_patch = Patch(color=colors_float["orange"], label=r'0.6 $\leq \sigma$ < 0.8')
green_patch = Patch(color=colors_float["green"], label=r'0.8 $\leq \sigma$ < 0.95')
super_green_patch = Patch(color=colors_float["super_green"], label=r'$\sigma \geq$  0.95')