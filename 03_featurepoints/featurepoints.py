# Nama: Sa'ad Abdul Hakim
# NIM: 13522092
# Fitur unik: Deteksi feature points dengan Harris, FAST, dan SIFT

from skimage import data, io, color, img_as_float, img_as_ubyte
from skimage.feature import corner_harris, corner_peaks, corner_fast, SIFT
import pandas as pd
import numpy as np


# Dataset
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
  gray_like = np.mean(img, axis=2) if img.ndim == 3 else img

  # HARRIS
  harris_resp = corner_harris(gray_like, method='k', k=0.05)
  coords_harris = corner_peaks(harris_resp, min_distance=5)

  marked_harris = color.label2rgb(
    np.zeros_like(gray_like, dtype=int),
    image=img,
    colors=['red'],
    alpha=0.7,
    bg_label=0
  )
  for y, x in coords_harris:
    yy, xx = np.clip(y, 0, img.shape[0] - 1), np.clip(x, 0, img.shape[1] - 1)
    marked_harris[int(yy), int(xx)] = [1, 0, 0]

  io.imsave(f"{name}_harris.png", img_as_ubyte(marked_harris))
  records.append(["Harris", name, len(coords_harris), float(np.mean(harris_resp)), float(np.max(harris_resp))])

  # FAST
  fast_resp = corner_fast(gray_like)
  coords_fast = corner_peaks(fast_resp, min_distance=5)

  marked_fast = color.label2rgb(
    np.zeros_like(gray_like, dtype=int),
    image=img,
    colors=['green'],
    alpha=0.7,
    bg_label=0
  )
  for y, x in coords_fast:
    yy, xx = np.clip(y, 0, img.shape[0] - 1), np.clip(x, 0, img.shape[1] - 1)
    marked_fast[int(yy), int(xx)] = [0, 1, 0]

  io.imsave(f"{name}_fast.png", img_as_ubyte(marked_fast))
  records.append(["FAST", name, len(coords_fast), float(np.mean(gray_like)), np.nan])

  # SIFT
  try:
    sift = SIFT()
    sift.detect_and_extract(gray_like)
    keypoints = sift.keypoints
    responses = sift.responses if hasattr(sift, "responses") else []

    marked_sift = color.label2rgb(
      np.zeros_like(gray_like, dtype=int),
      image=img,
      colors=['blue'],
      alpha=0.7,
      bg_label=0
    )
    for y, x in keypoints:
      yy, xx = np.clip(y, 0, img.shape[0] - 1), np.clip(x, 0, img.shape[1] - 1)
      marked_sift[int(yy), int(xx)] = [0, 0, 1]

    io.imsave(f"{name}_sift.png", img_as_ubyte(marked_sift))
    records.append([
      "SIFT", name, len(keypoints),
      float(np.mean(responses)) if len(responses) > 0 else np.nan,
      float(np.max(responses)) if len(responses) > 0 else np.nan
    ])
  except Exception as e:
    print(f"SIFT gagal pada {name}: {e}")
    records.append(["SIFT", name, 0, np.nan, np.nan])

# Simpan statistik ke CSV
df = pd.DataFrame(records, columns=["Metode", "Gambar", "Jumlah_Fitur", "Rata2_Response", "Maks_Response"])
df.to_csv("statistik_feature.csv", index=False)

print("Deteksi feature points selesai. Hasil tersimpan di folder 03_featurepoints.")
