"""
Here, all image features are implemented according to the specifications of Lan Luo's Management Science paper: Can Consumer-Posted Photos Serve as a Leading Indicator of Restaurant Survival? Evidence from Yelp
DOI: https://doi.org/10.1287/mnsc.2022.4359


Implementation details are saved in an Excel accessible via: https://tumde-my.sharepoint.com/:x:/g/personal/maximilian_konrad_tum_de/Eaxb2FNmiq5IrlhbbCYYcJ0BTIWDtR9A_HrEtSwsgXkKpg?e=7Q0tzM

Each feature will be an individual function.
1. Color Features:
   - Brightness
   - Saturation
   - Contrast
   - Clarity
   - Warm hue
   - Colorfulness

2. Composition Features:
   - Diagonal dominance
   - Rule of thirds
   - Physical visual balance
   - Color visual balance

3. Figure-ground Relationship Features:
   - Size difference
   - Color difference
   - Texture difference
   - Depth of field

"""