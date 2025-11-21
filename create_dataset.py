import cv2
import os
import glob

# Percorso del video
video_path = 'videos/video_20939_20230920_110603.mp4'  # Sostituisci con il tuo percorso

# Cartella dove salvare i fotogrammi
save_folder = 'train_data_raw/train_images'
os.makedirs(save_folder, exist_ok=True)

# Conta quanti file JPG ci sono nella cartella
existing_files = glob.glob(os.path.join(save_folder, 'frame_*.jpg'))
next_index = len(existing_files)

# Apri il video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Errore nell'apertura del video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fine del video o errore nella lettura.")
        break

    # Mostra il fotogramma
    cv2.imshow('Video', frame)

    key = cv2.waitKey(25)

    if key == ord('q'):
        break  # Premi 'q' per uscire
    elif key == ord(' '):  # Barra spaziatrice
        filename = os.path.join(save_folder, f"frame_{next_index:05d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Salvato: {filename}")
        next_index += 1

cap.release()
cv2.destroyAllWindows()
