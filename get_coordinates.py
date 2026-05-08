import cv2

# Carica l'immagine (o un frame del video)
img = cv2.imread('Sezione di strada.jpg')
punti = []

def click_event(event, x, y, flags, params):
    # Se clicchi con il tasto sinistro, salva il punto
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Punto campionato: [{x}, {y}]")
        punti.append((x, y))
        # Disegna un cerchietto dove hai cliccato
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Seleziona la ROI', img)

cv2.imshow('Seleziona la ROI', img)
cv2.setMouseCallback('Seleziona la ROI', click_event)

print("Clicca sui 4 angoli dell'area rossa. Premi un tasto qualsiasi per uscire.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("I tuoi punti sono:", punti)