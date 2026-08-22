import cv2  
import argparse

from smart_vision_event_filter.detection import detect_motion_boxes

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
        ret, frame = cap.read()
        if not ret:
            # Bei Videoende oder einem Lesefehler verlassen wir die Schleife.
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            # Der erste Frame dient nur als Referenz; erst ab Frame zwei kann eine
            # Veränderung zwischen zwei Zeitpunkten berechnet werden.
            previous_gray = gray
            continue

        motion_boxes = detect_motion_boxes(previous_gray, gray)
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