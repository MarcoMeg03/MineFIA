# MineFIA - Progetto per il corso di Fondamenti di Intelligenza Artificiale

## Introduzione
**MineFIA** è un progetto sviluppato nell'ambito del corso di Fondamenti di Intelligenza Artificiale, che utilizza la libreria [MineRL](https://minerl.io/) per addestrare agenti intelligenti tramite apprendimento per imitazione, con lo scopo di renderli in grado di compiere task complessi in ambienti Minecraft.<br>
Il nostro contributo all'interno di questo progetto è stato **ampliare la libreria originale** tramite lo sviluppo di script per aumentare i dati di addestramento, implementare un apprendimento per rinforzo che utilizza una funzione di reward creata da noi e con feedback umani, oltre che al definire completamente un nuovo task con insieme un nuovo ambiente `(FIA-Treechop-v0)` adatto al suo svolgimento.

## Funzionalità Principali
L'obiettivo e fare in modo che l'agente porti a termine le primissime fasi di gioco nella maniera più rapida e "razionale" possibile.
- **Definizione di un nuovo task** che include:
  - Raccogliere tronchi di legno di ogni tipo
  - Costruire assi di legno 

## Usa la nostra macchina docker
  Abbiamo predisposto per te una fantastica macchina docker<br>
  che ti eviterà di passare quello che abbiamo passato noi per utilizzare il framework MineRL!<br>
  Puoi trovarla [qui](https://hub.docker.com/r/giusopo/minefia).

## Installazione Manuale
  Ad ogni modo, se preferisci creare un tuo ambiente manualmente non ho niente in contrario!<br>
  Anzi, ecco dei semplici passi che dovrebbero guidarti nella creazione di un ambiente adatto a questo progetto.<br>
  
  Se vuoi avventurarti in una fantastica esperienza masochista, oppure vuoi fare un torto al tuo peggior nemico puoi provare ad installare il framework su Windows e/o MacOS.
  
- **Clona la Repository**:
  ```bash
    git clone https://github.com/MarcoMeg03/MineFIA.git
    cd MineFIA
  ```
  
- **Crea un ambiente virtuale:**<br>
  Puoi usare le versioni di python che vanno dalla 3.8 alla 3.10(consigliato), personalmente abbiamo scelto 3.10<br>
  (assicurati di usare questa versione quando creerai il tuo ambiente):
  ```bash
    sudo apt install python3.10
  ```
  Suggeriamo di utilizzare un ambiente virtuale per isolare le dipendenze del progetto
  ```bash
    python3 -m venv myenv
    source myenv/bin/activate
  ```
  
- **Installa le dipendenze richieste eseguendo:**
  ```bash
    pip install -r requirements.txt
  ```
  
- **Installa MineRL:**<br>
  Tieni conto che MineRL ha bisogno di Java 8:
  ```bash
    sudo apt install openjdk-8-jdk
   ```
  Ora puoi installare MineRL, il modo più semplice per farlo è tramite il repository ufficiale (v1.0.2)
   ```bash
    pip install git+https://github.com/minerllabs/minerl
   ```
  Adesso dovresti avere tutto ciò che ti serve per eseguire il progetto!
  
## Struttura del Progetto
Il progetto include i seguenti file principali:

### Training
- **`train.py`**: Script per addestrare gli agenti su vari task utilizzando Behavioural Cloning.
- **`behavioural_cloning.py`**: Implementazione dettagliata del processo di addestramento con gestione dello stato e ottimizzazioni.
- **`rf_learning.py`**: Implementazione dettagliata del processo di addestramento con reward automatiche.
- **`rl_human_feedback.py`**: Implementazione dettagliata del processo di addestramento con reward manuali.
- **`FIAenv.py`**: Implementazione dell'ambiente.

### Testing
Prima di eseguire l'agente è possibile impostare a `True` il booleano `TEST_x10_1500_STEP` e modificare a piacimento 
```python
if TEST_x10_1500_STEP:
    n_episodes = 10
    max_steps = 1500
```
In modo da eseguire una serie di episodi, sul terminale stamperà gli ogetti in possesso dell'agente al termine di ogni episodio.

Altri tipologie di test, come monitorare l'errore non si sono dimostrati metodi efficaci, siccome l'agente in un dato momento puo eseguire N azioni valide e diverse dal comportamento atteso, il metodo migliore per testare il modello è stato mediante un giudizio umano, sulla base degli oggetti ottenuti e sul quanto le sue azioni appaiano "ragionate".
```bash
 python run_agent.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show
```
### Altri File (COTS)
- **`data_loader.py`**: Caricamento dei dataset per il training e il testing.
- **`agent.py`**: Definizione della classe MineRLAgent per gestire osservazioni e azioni nell'ambiente.
- **`policy.py`**: Implementa la policy dell'agente basata su architettura Transformer.

## Esecuzione del Progetto
### Esecuzione di un modello pre addestrato
Una volta addestrato il tuo modello non ti basterà altro che eseguire `run_agent.py` nel seguente modo, specificando il modello da usare, i pesi da utilizzare e ambiente in cui svolgere il task (FIA-Treechop-v0), apparirà una finestra, e mediante una particolare versione di minecraft integrata in MineRL potrai vedere l'agente agire nel mondo di Minecraft:
```bash
 python run_agent.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show
```
### Training per imitazione
Per addestrare un agente per imitazione del comportamento esegui il seguente file come mostrato.<br>
Durante l'addestramento verra visualizzato sul terminale, e stasmpato su un file excel il valore della funzione di loss. così sarà possibile rendersi conto dell'andamento dell'addestramento in qualsiasi momento:<br>
(assicurati di aver impostato il modello e i pesi desiderati prima di avviare un addestramento, puoi<br>
farlo editando il file `train.py`)
```bash
python train.py
```

### Training per rinforzo
Come detto inizialmente abbiamo creato una funzione di reward ad hoc per il nostro task che attribuisce punteggi all'agente ogni volta che raccoglie uno di questi items:<br>
(anche in questo caso all'avvio dell'addestramento sarà presente un grafico che mostrerà l'andamento in tempo reale della reward attribuita, oltre l'effettiva finestra di gioco per visualizzare le azioni in tempo reale)
```python
MATERIAL_REWARDS = {
    "birch_log": 0.2,
    "dark_oak_log": 0.2,
    "jungle_log": 0.2,
    "oak_log": 0.2,
    "spruce_log": 0.2,
    "dark_oak_planks": 0.4,
    "jungle_planks": 0.4,
    "oak_planks": 0.4,
    "spruce_planks": 0.4,
    "crafting_table": 0.6, 
    "dirt": -0.01,
    "gravel": -0.01,
    "sand": -0.01
}
```
In modo che una volta appresi i comportamenti desiderati l'agente possa specializzarsi sempre di più in merito alla risoluzione del suo task e correggere alcuni comportamenti indesiderati:
```bash
 python rf_learning.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show --max-steps 2000 --episodes 10
```

### Training per rinforzo con feedback umani
Dove non è stato possibile automatizzare l'apprendimento per rinforzo tramite una reward automatica, abbiamo deciso di attribuire una reward manuale con feedback umani all'agente.<br> 
Questa costituisce l'ultima fase del training ed è stata utile ad eliminare uteriormente comportamenti indesiderati dell'agente e migliorare l'apprendimento.<br>
Si può attribuire una reward positiva con `+` e negativa con `-`.
```bash
python rl_human_feedback.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show --max-steps 5000 --episodes 7
```

## Dataset
Il progetto utilizza parzialmente il dataset BASALT di MineRL, sono stati estratti segmenti di alcuni video dal dataset originale in modo che fossero utili al raggiungimento del nostro task.<br>
Per un 40% i dati sono stati generati e specchiati da noi mediante gli appositi script.<br>

### Estrazione dati da dataset esistente
Utile ad estrarre i fotogrammi utili al nostro obiettivo dal dataset fornito da OpenAI.<br>
Sfrutta un semplice filtro che analizza il dataset iniziale e ne genera uno nuovo quando il soggetto<br>
del video conserva nell'inventario uno dei seguenti oggetti:<br>
```python
    useful_items = {
        "crafting_table","oak_planks", "birch_planks", "spruce_planks", 
        "jungle_planks", "acacia_planks", "dark_oak_planks",
        "oak_log", "birch_log", "spruce_log", "jungle_log", 
        "acacia_log", "dark_oak_log"
    }
```
Volendo è possibile cambiare il criterio con cui vengono tagliati i video cambiando questi items, per eseguire lo script invece:
```bash
python CutData.py
```
### Generazione osservazioni (.mp4 e JSONL)
Permette all'utente di sviluppare un breve gameplay nel mondo mi Minecraft tramite una finestra MineRL con lo scopo di generare una nuova osservazione e creare nuovi dati di addestramento, ciò è utile nella definizione di un nuovo task per cui non sono presenti dati.
```bash
python manual_recorder.py
```
### Mirroring dei dati
Per aumentare la quantità di dati abbiamo deciso di implementare uno script capace di specchiare i video e le azioni correlate ad ogni fotogramma,
non vengono specchiati fotogrammi che potrebbero causare problemi (es. non viene specchiato quando la GUI è aperta)
```bash
python MirrorData.py --input_folder ./path-videos-to-mirror --output_folder ./path-to-save-mirrored-videos
```
### Data Quality
Per verificare la qualità dei dati è stato creato lo script `visualize_mouse_movement.py`, ci ha permesso di accertarci che i movimenti del mouse registrati nel JSONL fossero coerenti con gli spostamenti effettivi nei video.
Semplicemente lo script mostra a video le coordinate riportate nel JSONL, eseguibile con:
```bash
python visualize_mouse_movement.py ./path-to-JSONL-file
```
## Licenza
Questo progetto è rilasciato sotto la licenza MIT. Consulta il file `LICENSE` per maggiori dettagli.

