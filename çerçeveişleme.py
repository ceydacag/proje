import cv2 #görüntü işleme 
import numpy as np #sayısal işlemler 
import matplotlib.pyplot as plt #grafik
import os #dosya işlemleri resimleri klasörüden okumak için 

#fotoğrafların dizini
frame_dir = "frames"


#tüm fotoğrafları listele
all_files = os.listdir(frame_dir)

#frame_ ile başlayanları al
frame_files = []
for file in all_files:
    if file.startswith("frame_"):
        frame_files.append(file)

#sıralı hale getir 
frame_files.sort()

#dosyaları birleştirdik
frames = []
for file in frame_files:
    frames.append(os.path.join(frame_dir, file))

#ilk fotoğrafı al ve griye çevir(akış diyagramları gri ölçekli görüntüler üzerinde daha iyi çalışırmış)
prev_frame = cv2.imread(frames[0])
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


"""
1- Shi-Tomasi Köşe Algılama Algoritması cv2.goodFeaturesToTrack
    Bir görüntüde "iyi takip edilebilecek" köşeleri (feature points) bulmak için geliştirilmiş bir köşe tespit algoritmasıdır.
Harris köşe algılama algoritmasının geliştirilmiş bir versiyonudur. Daha az gürültü içerir ve hareketi daha doğru takip etmeye yardımcı olur.
- Takip edilecek noktaları (köşeleri) belirlemek için kullanıldı.

2- Lucas-Kanade Optical Flow Algoritması cv2.calcOpticalFlowPyrLK
     İki ardışık frame (görüntü) arasındaki hareketi tahmin eder.
Özellik noktalarının (Shi-Tomasi ile bulduğumuz noktalar gibi) yeni konumlarını hesaplar.
Yerel (local) bir optik akış algoritmasıdır ve genellikle küçük hareketleri izlemek için kullanılır.
- Bu noktaların hareketini takip etmek için kullanıldı.
- Frame’ler arasında hareketi izlemek ve koordinatları güncellemek için.

"""

# Shi-Tomasi köşe algılama 
feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7) 

prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# eğer hiç nokta bulunmazsa 
if prev_points is None:
    print("Uyarı: İlk frame'de yeterli özellik noktası bulunamadı. Çıkılıyor...")
    exit()

# ilk bulunan noktaların ortalaması başlangıç noktası alınır
start_x, start_y = np.mean(prev_points, axis=0).flatten()
trajectory = [(start_x, start_y)] #hareket rotasının listesi

# Lucas-Kanade 
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.05))

# tüm görüntüleri griye çevir 
for i in range(1, len(frames)):
    frame = cv2.imread(frames[i])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ucas-Kanade yöntemi ile önceki noktaların yeni konumlarını bulunur
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

    if new_points is None or status is None:
        print(f"Uyarı: Frame {i} için optik akış hesaplanamadı. Önceki noktalar korunuyor...")
        continue

    # geçerli noktaları filtrele
    good_new = new_points[status == 1]
    good_old = prev_points[status == 1]

    if len(good_new) == 0:
        print(f"Uyarı: Frame {i} için iyi nokta bulunamadı. Yeni noktalar atanıyor...")
        prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        continue

    #hareket farkı
    dx = np.mean(good_new[:, 0] - good_old[:, 0])
    dy = np.mean(good_new[:, 1] - good_old[:, 1])

    # konum güncellemeleri
    scale_factor = 1.0  
    dx, dy = -dx * scale_factor, -dy * scale_factor  
    new_x = trajectory[-1][0] + dx
    new_y = trajectory[-1][1] + dy
    trajectory.append((new_x, new_y))

    # güncellenmiş değerleri atama
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

trajectory = np.array(trajectory)

# yumuşatma efektleri
trajectory[:, 0] = cv2.GaussianBlur(trajectory[:, 0].reshape(-1, 1), (3, 3), 0).flatten()
trajectory[:, 1] = cv2.GaussianBlur(trajectory[:, 1].reshape(-1, 1), (3, 3), 0).flatten()

#grafik
plt.figure(figsize=(8, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=3)  # Hareket çizgisi
plt.scatter(trajectory[0, 0], trajectory[0, 1], color='red', s=150, marker="s", label="Başlangıç Noktası")  # Başlangıç noktası
plt.title("Koordinat Grafiği")
plt.legend()
plt.xticks([])  # X eksenindeki değerleri gizle
plt.yticks([])  # Y eksenindeki değerleri gizle
plt.grid()
plt.show()
