# 🎨 Implémentation de Concept Sliders sur CryptoPunks

> Tentative de reproduction de la méthode "Concept Sliders" pour la génération conditionnelle de CryptoPunks.

## 🎯 Objectif du Projet

Ce projet vise à **reproduire la méthode "Concept Sliders"** décrite dans le papier :

> **Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models**  
> Gandikota et al., 2023 ([arXiv:2311.17216](https://arxiv.org/abs/2311.17216))

L'idée est d'apprendre des **vecteurs concepts** qui permettent de contrôler des attributs sémantiques (accessoires : casquette, pipe, cigarette, hoodie) dans un modèle de diffusion, sans réentraîner le modèle complet.

---

## 🏗️ Méthodologie Implémentée

### Étape 1 : Entraînement du DDPM de base

Entraînement d'un UNet pour prédire le bruit `ε` avec un concept vector nul (`c = 0`).

```
Loss : min_θ E||ε - ε_θ(x_t, t, c=0)||²
```

**Fichiers :**
- [src/model_vector.py](src/model_vector.py) : Architecture UNet avec injection de concept au bottleneck
- [src/train_ddpm.py](src/train_ddpm.py) : Script d'entraînement du DDPM de base
- [src/diffusion.py](src/diffusion.py) : Processus de diffusion (forward/reverse)

### Étape 2 : Apprentissage des Concept Vectors

Après avoir gelé le modèle DDPM pré-entraîné, on optimise un vecteur `c_k` pour chaque concept (accessoire) sur un sous-ensemble d'images filtré.

```
Loss : min_{c_k} E||ε - ε_θ(x_t, t, c_k)||²
```

L'injection se fait au bottleneck du UNet :
```
h' = h + α · c_k
```

où `h` est la représentation latente (512D) et `α` est un facteur d'échelle.

**Fichiers :**
- [src/train_concepts.py](src/train_concepts.py) : Optimisation des vecteurs concepts
- [src/create_subdatasets.py](src/create_subdatasets.py) : Création des sous-datasets par accessoire
- `concepts/` : Vecteurs concepts sauvegardés (`acc_cap.pt`, `acc_pipe.pt`, etc.)

### Étape 3 : Génération avec Combinaison de Concepts

À l'inférence, on combine linéairement les concepts :
```
c = Σ_k β_k · c_k
```

**Fichiers :**
- [src/generate_concepts.py](src/generate_concepts.py) : Génération avec injection de concepts
- [src/generate_with_concepts.py](src/generate_with_concepts.py) : Interface de génération

---

## ❌ Problème Rencontré : Échec de l'Apprentissage

### Observation

Pendant l'entraînement des concept vectors, **la norme des vecteurs `c_k` augmentait continuellement** au lieu de converger vers une représentation stable.

```
Epoch 1:   |c| = 0.05
Epoch 10:  |c| = 2.3
Epoch 50:  |c| = 15.7
Epoch 100: |c| = 45.2   ← divergence
```

### Analyse

Les vecteurs concepts n'arrivaient pas à apprendre correctement dans l'espace latent `h` du bottleneck :

1. **Espace non structuré** : Le modèle DDPM a été entraîné avec `c = 0`, donc l'espace latent n'a jamais été exposé à des variations de concepts. L'injection additive `h' = h + α·c` perturbe un espace qui n'a pas été conçu pour ça.

2. **Optimisation instable** : Sans contrainte, l'optimiseur pousse `||c_k||` → ∞ pour minimiser la loss, car augmenter la norme permet de "forcer" le modèle à produire des prédictions plus proches des images cibles.

3. **Pas de régularisation** : Contrairement au papier original qui utilise des LoRA (faible rang, régularisation implicite), notre approche avec un vecteur dense 512D n'a pas de contrainte structurelle.

4. **Distribution des features** : L'injection `h + α·c` peut faire sortir les features de leur distribution d'entraînement, causant des artefacts.

### Tentatives de correction (sans succès)

- Régularisation L2 sur `||c_k||`
- Diminution du learning rate
- Early stopping basé sur la norme
- Différentes valeurs de `α` (concept_scale)

---

## 📁 Structure du Projet

```
.
├── concepts/                       # Vecteurs concepts (tentative échouée)
│   ├── acc_cap.pt
│   ├── acc_cigarette.pt
│   ├── acc_hoodie.pt
│   └── acc_pipe.pt
├── models/
│   └── CRYPTOPUNKS/                # Modèle DDPM de base
│       └── cryptopunks1/
│           └── ckpt_final.pt
├── src/
│   ├── model_vector.py             # UNet avec injection concept
│   ├── train_ddpm.py               # Entraînement DDPM (c=0)
│   ├── train_concepts.py           # Apprentissage concepts (ÉCHEC)
│   ├── generate_concepts.py        # Génération avec concepts
│   ├── diffusion.py                # Process de diffusion
│   └── config.py                   # Configurations
└── runs/                           # Logs TensorBoard
```

---

## ⚙️ Configuration

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `T` | 1000 | Nombre de timesteps de diffusion |
| `img_size` | 32 | Taille des images (32×32) |
| `concept_dim` | 512 | Dimension des vecteurs concepts |
| `concept_scale` | 1.0 | Facteur α pour l'injection |
| `lr` (concepts) | 1e-3 | Learning rate pour l'apprentissage des concepts |
| `init_std` | 0.01 | Écart-type d'initialisation `c ~ N(0, σ²)` |

---

## 🚀 Utilisation

### 1. Entraîner le DDPM de base

```bash
python src/train_ddpm.py --config cryptopunks1 --epochs 100
```

### 2. Créer les sous-datasets par accessoire

```bash
python src/create_subdatasets.py
```

### 3. Apprendre un concept (résultats non satisfaisants)

```bash
python src/train_concepts.py \
    --checkpoint models/CRYPTOPUNKS/cryptopunks1/ckpt_final.pt \
    --concept cap \
    --epochs 100
```

### 4. Générer avec concepts

```bash
python src/generate_concepts.py \
    --checkpoint models/CRYPTOPUNKS/cryptopunks1/ckpt_final.pt \
    --concepts cap pipe \
    --weights 1.0 0.5 \
    --n 4
```

---

## 📚 Références

- [Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models (Gandikota et al., 2023)](https://arxiv.org/abs/2311.17216)
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [CryptoPunks Dataset](https://www.kaggle.com/datasets/chwasiq0569/cryptopunks-pixel-art-dataset)

---

## ✅ Approche Alternative (Fonctionnelle)

Face à l'échec de la méthode Concept Sliders, une approche alternative a été implémentée : **entraîner le modèle directement avec conditionnement sur les accessoires**.

- [src/model_conditioned.py](src/model_conditioned.py) : UNet avec embedding learnable (multi-hot → concept 512D)
- [src/train_ddpm_conditioned.py](src/train_ddpm_conditioned.py) : Entraînement end-to-end avec CFG dropout
- [app.py](app.py) : Interface Streamlit pour la génération

Cette approche fonctionne car le modèle apprend dès le départ à utiliser les vecteurs d'accessoires, plutôt que d'essayer d'injecter des concepts dans un espace non préparé.

```bash
# Entraînement conditionné
python src/train_ddpm_conditioned.py --epochs 50

# Interface de génération
streamlit run app.py
```

---

## 📄 Licence

MIT License

## 📄 Licence

MIT License - voir [LICENSE](LICENSE) pour plus de détails.

---

## 👤 Auteur

Projet Deep Learning - EPITA S9 (2025-2026)
