# Medical Image PEFT Classifier

Interpretowalny klasyfikator obrazów medycznych: fine-tuning foundation modelu
metodą LoRA, wyjaśnialność (Grad-CAM), niepewność predykcji, wdrożenie (API + demo).

## Plan (4 tygodnie)

**Tydzień 1 — dane i baseline**
- EDA, podział train/val/test (uwaga na class imbalance)
- Baseline: full fine-tuning małego modelu (np. ResNet/ViT-Small)
- Tracking eksperymentów (W&B lub MLflow)

**Tydzień 2 — PEFT**
- Fine-tuning foundation modelu (np. DINOv2, CLIP) z LoRA (`peft`)
- Porównanie: full fine-tuning vs. LoRA vs. linear probing
  (accuracy, liczba trenowalnych parametrów, czas treningu)

**Tydzień 3 — wyjaśnialność i niepewność**
- Grad-CAM (`pytorch-grad-cam`) i/lub Attention Rollout dla ViT
- Estymacja niepewności: MC Dropout albo Deep Ensembles

**Tydzień 4 — wdrożenie**
- FastAPI: endpoint `/predict` (obraz -> klasa + pewność)
- Streamlit: proste demo UI
- Docker + docker-compose
- CI (GitHub Actions): lint + testy

## Struktura repo

```
medical-image-peft/
├── configs/default.yaml     # konfiguracja (dataset, model, LoRA, trening)
├── src/
│   ├── data/dataset.py       # Dataset + DataLoadery
│   ├── models/peft_model.py   # backbone + LoRA/full/linear_probe
│   ├── training/train.py       # pętla treningowa
│   ├── explainability/gradcam.py # Grad-CAM / Attention Rollout
│   └── utils/logging.py        # logger, tracking eksperymentów
├── api/                    # FastAPI (main.py, schemas.py)
├── app/streamlit_app.py    # demo UI
├── tests/                  # testy jednostkowe
├── notebooks/               # EDA i eksperymenty
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Sugerowany dataset

HAM10000 (klasyfikacja zmian skórnych) — mały, dobrze udokumentowany,
dostępny na Kaggle. Alternatywa: CheXpert / ChestX-ray14 (RTG klatki piersiowej).

## Wyniki (uzupełnij w trakcie pracy)

| Metoda           | Trenowalne parametry | Accuracy | F1 |
|------------------|----------------------|----------|----|
| Full fine-tuning |                      |          |    |
| LoRA             |                      |          |    |
| Linear probing   |                      |          |    |
