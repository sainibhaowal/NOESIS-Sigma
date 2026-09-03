from PIL import Image

img = Image.open(
    '/home/sephi-asi/NOESIS-Σ/Frontend/src/components/Landing/noesis_sigma_logo_concept.png'
)
print('Mode:', img.mode)
print('Size:', img.size)

if img.mode == 'RGBA':
    data = list(img.getdata())
    transparent_count = sum(1 for p in data if p[3] < 255)
    print(f'Transparent pixels: {transparent_count} / {len(data)}')
else:
    print('Image does not have an alpha channel (no transparency)')
