# Nama: Sa'ad Abdul Hakim
# NIM: 13522092
# Fitur unik: Filtering dengan Gaussian & Median Filter

from skimage import data, io, img_as_ubyte
from skimage.filters import gaussian, median
from skimage.morphology import disk
import pandas as pd
import numpy as np

# Load Dataset
images = {
  "cameraman": data.camera(),
  "coins": data.coins(),
  "checkerboard": data.checkerboard(),
  "astronaut": data.astronaut(),
  "personal": io.imread("../Candy.jpg")
}

# Parameter Filter
params = []

for name, img in images.items():
  # Gaussian Filter
  gaussian_img = gaussian(img, sigma=1.5)
  io.imsave(f"{name}_gaussian.png", img_as_ubyte(gaussian_img))
  params.append(["Gaussian", name, "sigma=1.5"])

  # Median Filter
  if img.ndim == 3:  # RGB
    median_img = np.zeros_like(img)
    for c in range(3):
      median_img[:, :, c] = median(img[:, :, c], disk(5))
  else:
    median_img = median(img, disk(5))
  io.imsave(f"{name}_median.png", img_as_ubyte(median_img))
  params.append(["Median", name, "radius=5"])

# Simpan Parameter
df = pd.DataFrame(params, columns=["Filter", "Gambar", "Parameter"])
df.to_csv("parameter.csv", index=False)

print("Filtering selesai. Hasil tersimpan di folder 01_filtering.")
