import cv2  # OpenCV-Bibliothek: Lesen, Verarbeiten und Anzeigen von Video-Frames.
import argparse


THRESHOLD_VALUE = 60  # Mindest-Helligkeitsänderung, die als Bewegung zählt.
DILATE_ITERATIONS = 5  # Verbindet nahe beieinanderliegende Bewegungs-Pixel.
MIN_AREA = 8000  # Kleinere Flächen gelten als Rauschen und werden ignoriert.
FRAME_DELAY = 1  # Wartezeit für die Tastatureingabe in Millisekunden.


def main():

    parser = argparse.ArgumentParser(
        description="recognices movement in video file"
    )
    parser.add_argument(
        "video_path",
        help="path to video file"
    )

    args = parser.parse_args()


    # Video-Datei öffnen. `cap` steht für "capture" und ist das Objekt, das den
    # Zugriff auf die einzelnen Bilder (Frames) des Videos verwaltet.
    video_path = args.video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Ohne geöffnetes Video kann die Verarbeitung nicht sinnvoll starten.
        print(f"Could not open video file at: {video_path}")
        raise SystemExit


    # Zu Beginn gibt es noch kein vorheriges Bild, mit dem wir vergleichen könnten.
    previous_gray = None

    # Das Video wird Frame für Frame verarbeitet, bis es endet oder `q` gedrückt wird.
    while True:
        # `ret` ist True, wenn ein Frame erfolgreich gelesen wurde.
        # `frame` ist das gelesene Farbbild als NumPy-Array im BGR-Farbformat.
        ret, frame = cap.read()
        if not ret:
            # Bei Videoende oder einem Lesefehler verlassen wir die Schleife.
            break

        # Aus BGR (Blau-Grün-Rot) wird ein Graustufenbild. Für die reine
        # Bewegungsanalyse reicht ein Helligkeitswert pro Pixel statt drei Farbwerte.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            # Der erste Frame dient nur als Referenz; erst ab Frame zwei kann eine
            # Veränderung zwischen zwei Zeitpunkten berechnet werden.
            previous_gray = gray
            continue

        # Pro Pixel wird die absolute Helligkeitsdifferenz zwischen dem vorherigen
        # und dem aktuellen Frame berechnet. Große Differenzen deuten auf Bewegung hin.
        diff = cv2.absdiff(previous_gray, gray)

        # Das Graustufen-Differenzbild wird binär: Werte ab THRESHOLD_VALUE werden
        # weiß (255), alle kleineren Werte schwarz (0). Der erste Rückgabewert ist
        # der verwendete Schwellenwert und wird hier nicht benötigt (`_`).
        _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # Dilation vergrößert weiße Bereiche leicht. Das verbindet kleine Lücken in
        # einer Bewegung, kann bei zu vielen Iterationen aber auch Rauschen vergrößern.
        dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)

        # Konturen sind Umrisse zusammenhängender weißer Bereiche. RETR_EXTERNAL
        # berücksichtigt nur äußere Umrisse; innere Löcher werden nicht separat gezählt.
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Hier sammeln wir die Rechtecke aller ausreichend großen Bewegungen dieses Frames.
        motion_boxes = []

        for contour in contours:
            # Die Fläche wird in Pixeln gemessen und hilft, kleine Störungen zu filtern.
            area = cv2.contourArea(contour)

            if area < MIN_AREA:
                # `continue` überspringt nur diese Kontur und prüft die nächste.
                continue

            # Das kleinste achsenparallele Rechteck um die Kontur:
            # x/y = linke obere Ecke, w/h = Breite/Höhe.
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))

            # Einzelne Bewegungsregionen werden blau eingezeichnet. OpenCV verwendet
            # BGR, daher ist (255, 0, 0) blau und nicht rot.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if motion_boxes:
            # Aus allen Einzelrechtecken wird ein gemeinsames Gesamt-Rechteck gebildet.
            # `min` bestimmt die am weitesten links/oben liegenden Kanten, `max` die
            # am weitesten rechts/unten liegenden Kanten.
            min_x = min(x for x, y, w, h in motion_boxes)
            min_y = min(y for x, y, w, h in motion_boxes)
            max_x = max(x + w for x, y, w, h in motion_boxes)
            max_y = max(y + h for x, y, w, h in motion_boxes)

            # Das gemeinsame Rechteck und der Text machen den Frame für Menschen lesbar.
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "Motion Detected",
                (min_x, min_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Aktuellen, bereits markierten Frame im Fenster anzeigen.
        cv2.imshow("Motion Detection", frame)

        # Maximal FRAME_DELAY Millisekunden auf eine Taste warten. `& 0xFF` behält
        # nur die unteren acht Bits des Tastencodes; das macht den Vergleich auf
        # verschiedenen OpenCV-Plattformen zuverlässig. `ord("q")` liefert den
        # numerischen Zeichencode der Taste q.
        if cv2.waitKey(FRAME_DELAY) & 0xFF == ord("q"):
            break

        # Erst am Ende wird der aktuelle Frame zur Referenz für die nächste Runde.
        previous_gray = gray

    # Ressourcen aufräumen: Video-Datei freigeben und das OpenCV-Fenster schließen.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()