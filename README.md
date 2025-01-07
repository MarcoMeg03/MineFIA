# MineFIA - Progetto corse per il corso di Fondamenti di Intelligenza Artificiale

## Introduzione
**MineFIA** è un progetto sviluppato nell'ambito del corso di Fondamenti di Intelligenza Artificiale, che utilizza la libreria [MineRL](https://minerl.io/) per creare agenti AI in grado di completare task complessi in ambienti Minecraft. Questo progetto si concentra sull'addestramento tramite Behavioural Cloning,Reinforcement learning con l'aggiunta di funzioni di reward che includono informazioni dall'inventario e Renforcement learning con feedback umani.

## Funzionalità Principali
- **Supporto per Ambienti BASALT**: Task supportati includono:
  - Raccogliere tronchi di legno
  - Costruire assi di legno 
  - Costruire un banco da lavoro
  - Costruire una cesta dove riporre il legno

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

### Testing
- **`run_agent.py`**: Esegue agenti pre-addestrati in ambienti specificati.

### Altri File
- **`data_loader.py`**: Caricamento dei dataset per il training e il testing.
- **`agent.py`**: Definizione della classe MineRLAgent per gestire osservazioni e azioni nell'ambiente.
- **`policy.py`**: Implementa la policy dell'agente basata su architettura Transformer.

## Esecuzione del Progetto
### Training
Per addestrare un agente su un task specifico, esegui:
```bash
python train.py
```

### Testing
Per testare un agente addestrato, utilizza uno degli script di test:
```bash
python run_agent.py --models --wheight
``` 

## Dataset
Il progetto utilizza parzialmente il dataset BASALT di MineRL, sono stati estratti segmenti di alcuni video dal dataset originale.
per un 40% i dati sono stati generati e specchiati da noi mediante gli appositi script.


## Licenza
Questo progetto è rilasciato sotto la licenza MIT. Consulta il file `LICENSE` per maggiori dettagli.

