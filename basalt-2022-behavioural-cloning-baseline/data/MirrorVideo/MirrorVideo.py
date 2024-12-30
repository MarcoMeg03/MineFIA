import cv2
import os
import glob
import json

def mirror_video(input_path, output_path):
    # Apri il video originale
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {input_path}")
        return

    # Ottieni le proprietà del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per output in formato .mp4

    if fps == 0 or width == 0 or height == 0:
        print(f"Errore: il video {input_path} ha FPS o dimensioni non validi.")
        cap.release()
        return

    # Configura il writer per salvare il video specchiato
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        count += 1
        # Specchia il frame orizzontalmente
        mirrored_frame = cv2.flip(frame, 1)
        # Scrivi il frame nel nuovo video
        out.write(mirrored_frame)

    # Chiudi tutto
    cap.release()
    out.release()
    print(f"Video specchiato salvato in: {output_path}")

def mirror_json(input_json_path, output_json_path):

    #Trova la larghezza del video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    with open(input_json_path, "r") as infile:
        lines = infile.readlines()

    mirrored_lines = []
    for line in lines:
        data = json.loads(line)

        # Specchia il movimento della telecamera o del mouse
        if "mouse" in data:
            # Inverti i movimenti orizzontali
            data["mouse"]["dx"] = -data["mouse"]["dx"]  # Inverti la direzione X
            data["mouse"]["scaledX"] = -data["mouse"]["scaledX"]  # Inverti scaledX
            data["mouse"]["x"] = width - data["mouse"]["x"]  # Inverti posizione X rispetto alla larghezza dello schermo (assumendo 1280 come larghezza)

        # Scambia i movimenti sinistra/destra se necessario
        if "keyboard" in data and "keys" in data["keyboard"]:
            keys = data["keyboard"]["keys"]
            if "key.keyboard.a" in keys or "key.keyboard.d" in keys:
                # Scambia i movimenti sinistra/destra
                new_keys = []
                new_chars = []  # Per aggiornare il valore di chars
                for key in keys:
                    if key == "key.keyboard.a":
                        new_keys.append("key.keyboard.d")
                        new_chars.append("d")
                    elif key == "key.keyboard.d":
                        new_keys.append("key.keyboard.a")
                        new_chars.append("a")
                    else:
                        new_keys.append(key)
                        if "chars" in data["keyboard"]:  # Mantieni i caratteri invariati per altri tasti
                            new_chars.append(data["keyboard"]["chars"])

                # Aggiorna keys e chars
                data["keyboard"]["keys"] = new_keys
                if "chars" in data["keyboard"]:
                    data["keyboard"]["chars"] = "".join(new_chars)  # Combina i caratteri in una stringa
        
        # Specchia la hotbar
        if "hotbar" in data:
            data["hotbar"] = 8 - data["hotbar"]

        # Aggiungi la riga specchiata
        mirrored_lines.append(json.dumps(data))

    # Scrivi il nuovo file JSONL
    with open(output_json_path, "w") as outfile:
        outfile.write("\n".join(mirrored_lines) + "\n")
    print(f"JSONL specchiato salvato in: {output_json_path}")

# Se si vuole specchiare dei dati di addestramento
# modificare input e output folder

input_folder = "../MineRLBasaltFindCave-v0"  
output_folder = "../MineRLBasaltFindCave-v0-Mirrored"

os.makedirs(output_folder, exist_ok=True)

# Trova tutti i file video nella cartella
video_paths = glob.glob(os.path.join(input_folder, "*.mp4"))

for video_path in video_paths:
    video_name = os.path.basename(video_path)
    mirrored_video_path = os.path.join(output_folder, f"mirrored_{video_name}")

    # Specchia il video
    mirror_video(video_path, mirrored_video_path)

    # Specchia il JSONL corrispondente
    json_name = video_name.replace(".mp4", ".jsonl")
    input_json_path = os.path.join(input_folder, json_name)
    mirrored_json_path = os.path.join(output_folder, f"mirrored_{json_name}")

    if os.path.exists(input_json_path):
        mirror_json(input_json_path, mirrored_json_path)
    else:
        print(f"Attenzione: JSONL non trovato per {video_name}")
