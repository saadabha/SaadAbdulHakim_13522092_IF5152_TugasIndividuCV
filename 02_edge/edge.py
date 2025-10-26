# Nama: Sa'ad Abdul Hakim
# NIM: 13522092
# Fitur unik: Edge detection dengan Sobel & Canny

from skimage import data, io, img_as_float, img_as_ubyte
from skimage.filters import sobel
from skimage.feature import canny
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

records = []

for name, img in images.items():
  img = img_as_float(img)

  # Simpan gambar asli (before)
  io.imsave(f"{name}_original.png", img_as_ubyte(img))

  # Sobel Edge Detection
  if img.ndim == 3:  # RGB
    sobel_edges = np.zeros_like(img)
    for c in range(3):
      sobel_edges[:, :, c] = sobel(img[:, :, c])
  else:
    sobel_edges = sobel(img)

  io.imsave(f"{name}_sobel.png", img_as_ubyte(sobel_edges))
  records.append([
    "Sobel",
    name,
    "—",
    "—",
    "Sampling per channel" if img.ndim == 3 else "Single channel",
    "Gradien lokal, hasil halus tapi tepi tebal"
  ])

  # Canny Edge Detection
  gray_like = np.mean(img, axis=2) if img.ndim == 3 else img

  # low
  edge_canny_low = canny(gray_like, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
  io.imsave(f"{name}_canny_low.png", img_as_ubyte(edge_canny_low))
  records.append([
    "Canny (Low)", name, "10", "50",
    "Menangkap lebih banyak detail, namun lebih noisy"
  ])

  # high
  edge_canny_high = canny(gray_like, sigma=3.0, low_threshold=0.1, high_threshold=0.3)
  io.imsave(f"{name}_canny_high.png", img_as_ubyte(edge_canny_high))
  records.append([
    "Canny (High)", name, "50", "150",
    "Mengabaikan tepi lemah, hasil lebih bersih tapi kehilangan detail halus"
  ])

# Simpan tabel threshold & efek sampling
df = pd.DataFrame(records, columns=[
  "Metode", "Gambar", "Low_Threshold", "High_Threshold",
  "Parameter_Sampling", "Efek_Hasil"
])
df.to_csv("threshold.csv", index=False)

print("Edge detection selesai. Hasil tersimpan di folder 02_edge.")
