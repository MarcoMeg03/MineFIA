# MineFIA - Progetto corse per il corso di Fondamenti di Intelligenza Artificiale

## Introduzione
**MineFIA** è un progetto sviluppato nell'ambito del corso di Fondamenti di Intelligenza Artificiale, che utilizza la libreria [MineRL](https://minerl.io/) per creare agenti AI in grado di completare task complessi in ambienti Minecraft. Questo progetto si concentra sull'addestramento tramite Behavioural Cloning,Reinforcement learning con l'aggiunta di funzioni di reward che includono informazioni dall'inventario e Renforcement learning con feedback umani.

## Funzionalità Principali
- **Supporto per Ambienti BASALT**: Task supportati includono:
  - Raccogliere tronchi di legno
  - Costruire assi di legno 
  - Costruire un banco da lavoro

## Prerequisiti
- Python 3.8 o superiore
- Pytorch
- MineRL
- OpenAI VPT

Installa le dipendenze richieste eseguendo:
```bash
pip install -r requirements.txt
```

## Struttura del Progetto
Il progetto include i seguenti file principali:

### Training
- **`train.py`**: Script per addestrare gli agenti su vari task utilizzando Behavioural Cloning.
- **`behavioural_cloning.py`**: Implementazione dettagliata del processo di addestramento con gestione dello stato e ottimizzazioni.
- **`rf_learning.py`**: Implementazione dettagliata del processo di addestramento con reward automatiche.
- **`rl_human_feedback.py`**: Implementazione dettagliata del processo di addestramento con reward manuali.
- **`FIAenv.py`**: Implementazione del ambiente.

### Testing
- **`run_agent.py`**: Esegue agenti pre-addestrati in ambienti specificati.
```bash
 python run_agent.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show
```
### Altri File (COTS)
- **`data_loader.py`**: Caricamento dei dataset per il training e il testing.
- **`agent.py`**: Definizione della classe MineRLAgent per gestire osservazioni e azioni nell'ambiente.
- **`policy.py`**: Implementa la policy dell'agente basata su architettura Transformer.

## Esecuzione del Progetto
### Esecuzione di un modello pre addestrato
- **`run_agent.py`**: Esegue agenti pre-addestrati in ambienti specificati.
```bash
 python run_agent.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show
```
### Training per imitazione
Per addestrare un agente per imitazione del comportamento su un task specifico, esegui:
```bash
python train.py
```

### Training per rinforzo
```bash
 python rf_learning.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show --max-steps 5000 --episodes 7
```

### Training per rinforzo con feedback umani
cliccando "+" e "-" si possono dare reward al modello in fase di esecuzione
```bash
python rl_human_feedback.py --weights ./train/MineRLBasaltFindWood.weights --model ./data/VPT-models/foundation-model-1x.model --env FIA-Treechop-v0 --show --max-steps 5000 --episodes 7
```

## Dataset
Il progetto utilizza parzialmente il dataset BASALT di MineRL, sono stati estratti segmenti di alcuni video dal dataset originale.
per un 40% i dati sono stati generati e specchiati da noi mediante gli appositi script.
### Estrazione dati da dataset esistente
Utile ad estrarre i fotogrammi utili al nostro obiettivo dal dataset fornito da OpenAI
```bash
python CutData.py
```
### Generazione osservazioni
Utile a registrare nuove osservazioni (.mp4 e JSONL)
```bash
python manual_recorder.py
```
### Mirroring dei dati
Per aumentare la quantità di dati abbiamo deciso di implementare uno script capace di specchiare i video e le azioni correlate ad ogni fotogramma,
non vengono specchiati fotogrammi che potrebbero causare problemi(non viene specchiato quando la GUI è aperta)
```bash
python MirrorData.py --input_folder ./path --output_folder
```

## Licenza
Questo progetto è rilasciato sotto la licenza MIT. Consulta il file `LICENSE` per maggiori dettagli.

