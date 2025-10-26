# Nama: Sa'ad Abdul Hakim
# NIM: 13522092
# Fitur unik: Simulasi kalibrasi sederhana dengan checker

from skimage import data, color, io, img_as_ubyte, img_as_float
from skimage.transform import resize, ProjectiveTransform, warp
import numpy as np
import pandas as pd

# Checkerboard sebagai pola kalibrasi
checker = data.checkerboard()
checker = resize(checker, (300, 300), anti_aliasing=True)
rows, cols = checker.shape

# Titik referensi dunia
src_points = np.array([
  [0, 0],
  [cols - 1, 0],
  [cols - 1, rows - 1],
  [0, rows - 1]
])

# Titik tujuan - disimulasikan efek kamera miring
dst_points = np.array([
  [cols * 0.15, rows * 0.1],
  [cols * 0.95, rows * 0.05],
  [cols * 0.85, rows * 0.95],
  [cols * 0.05, rows * 0.85]
])

# Estimasi homografi
tform = ProjectiveTransform()
tform.estimate(src_points, dst_points)

# Terapkan ke checkerboard untuk verifikasi
warped_checker = warp(checker, tform.inverse)
checker_rgb = color.gray2rgb(checker)
warped_checker_rgb = color.gray2rgb(warped_checker)
overlay_checker = 0.6 * checker_rgb + 0.4 * warped_checker_rgb

# Simpan hasil checkerboard
io.imsave("checker_transformed.png", img_as_ubyte(warped_checker_rgb))
io.imsave("checker_overlay.png", img_as_ubyte(overlay_checker))

# Terapkan hasil transformasi ke semua gambar uji
images = {
  "cameraman": data.camera(),
  "coins": data.coins(),
  "astronaut": data.astronaut(),
  "personal": io.imread("../Candy.jpg")
}

records = []

for name, img in images.items():
  img = img_as_float(img)
  img = resize(img, (300, 300), anti_aliasing=True)

  if img.ndim == 2:
    img = color.gray2rgb(img)

  # Terapkan homografi hasil kalibrasi checkerboard
  warped = warp(img, tform.inverse, output_shape=(300, 300))
  overlay = 0.5 * img + 0.5 * warped

  # Simpan hasil
  io.imsave(f"{name}_transformed.png", img_as_ubyte(warped))
  io.imsave(f"{name}_overlay.png", img_as_ubyte(overlay))

  # Simpan data ke tabel
  records.append([
    name,
    np.round(tform.params[0, 0], 5), np.round(tform.params[0, 1], 5), np.round(tform.params[0, 2], 5),
    np.round(tform.params[1, 0], 5), np.round(tform.params[1, 1], 5), np.round(tform.params[1, 2], 5),
    np.round(tform.params[2, 0], 6), np.round(tform.params[2, 1], 6), np.round(tform.params[2, 2], 5)
  ])

# Simpan matriks homografi hasil estimasi
columns = [
  "Gambar",
  "h00", "h01", "h02",
  "h10", "h11", "h12",
  "h20", "h21", "h22"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("homography_matrix.csv", index=False)

print("Simulasi kalibrasi selesai. Hasil tersimpan di folder 04_geometry.")