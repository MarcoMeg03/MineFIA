import os
import json
import cv2

def is_inventory_useful(inventory):
    """
    Verifica se l'inventario contiene solo oggetti utili.
    """
    useful_items = {"crafting_table", "birch_planks", "oak_planks", "birch_log", "oak_log"}
    return all(item["type"] in useful_items for item in inventory)


def get_cut_timestamp(jsonl_path):
    """
    Legge un file JSONL e restituisce il tick e il tempo relativo in cui tagliare il video.
    """
    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    last_useful_tick = None
    last_useful_time = None
    first_milli = None
    average_server_tick_duration = 0

    for line in lines:
        data = json.loads(line)

        # Calcola il tempo relativo al primo frame
        if first_milli is None:
            first_milli = data["milli"]

        # Calcola la durata media del tick
        server_tick_duration = data.get("serverTickDurationMs", 50.0)  # Default 50ms
        average_server_tick_duration = (average_server_tick_duration + server_tick_duration) / 2

        # Verifica se l'inventario è utile
        inventory = data.get("inventory", [])
        if not is_inventory_useful(inventory):
            print(f"Inventario non utile trovato: {inventory}")

            # Salva l'ultimo tick utile
            last_useful_tick = data["tick"]
            last_useful_time = (data["tick"] * average_server_tick_duration) / 1000.0  # Tempo relativo in secondi
            print(f"Inventario utile a tick {last_useful_tick}: {inventory}")

            break
        
    print(f"Ultimo tick utile: {last_useful_tick}, tempo relativo: {last_useful_time}")
    return last_useful_tick, last_useful_time


def trim_video(video_path, output_path, cut_time):
    """
    Taglia il video fino al punto specificato.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return

    # Ottieni le proprietà del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcola il numero di frame da tagliare
    cut_frame = int(cut_time * fps)
    print(f"FPS: {fps}, Frame totali: {total_frames}, Frame di taglio: {cut_frame}")

    # Configura il writer per il video tagliato
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret or current_frame >= cut_frame:
            break

        out.write(frame)
        current_frame += 1

    # Rilascia risorse
    cap.release()
    out.release()
    print(f"Video tagliato salvato in: {output_path}")

    return cut_frame


def trim_jsonl(jsonl_path, output_jsonl_path, max_tick):
    """
    Taglia il file JSONL fino al tick specificato.
    """
    with open(jsonl_path, "r") as infile:
        lines = infile.readlines()

    trimmed_lines = []
    for line in lines:
        data = json.loads(line)
        if data["tick"] > max_tick:
            break
        trimmed_lines.append(line)

    with open(output_jsonl_path, "w") as outfile:
        outfile.writelines(trimmed_lines)

    print(f"JSONL tagliato salvato in: {output_jsonl_path}")


def process_videos(input_folder, output_folder):
    """
    Processa i video e i relativi JSONL nella cartella di input.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Trova tutti i file video e JSONL
    video_paths = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    for video_name in video_paths:
        jsonl_name = video_name.replace(".mp4", ".jsonl")
        video_path = os.path.join(input_folder, video_name)
        jsonl_path = os.path.join(input_folder, jsonl_name)

        if not os.path.exists(jsonl_path):
            print(f"JSONL non trovato per {video_name}, salto...")
            continue

        # Ottieni il tick e il tempo di taglio
        last_tick, cut_time = get_cut_timestamp(jsonl_path)

        if cut_time is None or last_tick is None:
            print(f"Nessun taglio necessario per {video_name}, salto...")
            continue

        # Taglia il video e salva nella cartella di output
        output_video_path = os.path.join(output_folder, f"trimmed_{video_name}")
        cut_frame = trim_video(video_path, output_video_path, cut_time)

        # Taglia il JSONL e salva nella cartella di output
        output_jsonl_path = os.path.join(output_folder, f"trimmed_{jsonl_name}")
        trim_jsonl(jsonl_path, output_jsonl_path, last_tick)

if __name__ == "__main__":
    input_folder = "../DataTest"
    output_folder = "../CuttedVideos"
    process_videos(input_folder, output_folder)
