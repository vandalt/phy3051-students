# Huggingface

Dans le notebook précédent, nous avons vu comment un transformer peut-être créé et modifié directement avec PyTorch.
On aurait également pu aller un pas plus loin et implémenter le modèle nous même avec PyTorch.
C'est un exercice intéressant, mais qui demande un peu plus de temps que ce que nous avons dans les TPs.
Ceci dit, je vous encourage à consulter des ressources en ligne à ce sujet si ça vous intéresse.

Aujourd'hui, nous explorerons plutôt [_huggingface_](https://huggingface.co/).
Huggingface est un peu comme un GitHub pour les modèles d'IA.
Les utilisateurs peuvent publier l'architecture de leur modèle, des poids pré-entraînés et des ensembles de données.
Le site fourni aussi des librairies implémentant certains communs, notamment des [transformers](https://huggingface.co/docs/transformers/index) et des [modèles de diffusion](https://huggingface.co/docs/diffusers/index).
Voir la section [documentation](https://huggingface.co/docs) pour plus de détails.
Dans ce notebook, nous allons nous familiariser avec la librairie [transformers](https://huggingface.co/docs/transformers/index).

La librairie `transformers` va un peu dans la direction opposée d'une implémentation complète en PyTorch: presque toutes les opérations sont cachées derrières des classes nous permettant simplement de spécifier les paramètres de notre modèle.
Ce n'est pas la meilleure façon de comprendre tous les détails d'un modèle, mais c'est pratique pour le tester rapidement et comprendre comment il interprète les données.
De plus, tous les modèles de la librairie sont disponible [sur GitHub](https://github.com/huggingface/transformers/tree/main/src/transformers/models).
N'hésitez pas à les consulter!

## Installation

Pour utiliser `transformers`, il faudra d'abord l'installer.

```bash
python -m pip install transformers
```

Si vous n'avez pas de GPU, vous pouvez remplacer `transformers` par `transformers[torch]`.

## Connexion

L'accès à certains modèles Huggingface requiert un compte et une authentification.
Pour se connecter, on peut utiliser `notebook_login()` dans un notebook et `login()` dans un terminal

```python
def is_notebook() -> bool:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__ 
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

NOTEBOOK = is_notebook()
```

```python
from huggingface_hub import notebook_login, login

if NOTEBOOK:
    notebook_login()
else:
    login()
```

## Pipelines

L'interface la plus simple de `transformers` est la classe `Pipeline`.
Celle-ci nous permet d'importer et d'utiliser un transformeur en trois lignes de code!

### Génération de texte

Bien que ce ne soit pas de l'analyse de données physique, la génération de texte est tellement omni-présente dans les dernières années qu'il peut être intéressant de voir comment l'appliquer avec Huggingfance.

```python
from transformers import pipeline

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is Markov Chain Monte Carlo"},]
        },
    ],
]

txt_pipeline = pipeline(task="text-generation", model="google/gemma-3-1b-it")
# txt_pipeline = pipeline(task="text-generation")
```

```python
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is Markov Chain Monte Carlo? Explain in two sentences."},]
        },
    ],
]
# messages = "Markov Chain Monte Carlo is an inference method that"
```

```python
simple = isinstance(messages, str)
reply = txt_pipeline(messages, max_new_tokens=100)
if simple:
    print(reply[0]["generated_text"])
else:
    print(reply[0][0]["generated_text"][2]["content"])
```

```python
del txt_pipeline
```

### Classification d'image

L'interface `pipeline` ne se limite bien sûr pas à la génération de texte.
On peut spécifier une autre tâche via le premier argument, `task`.
Par exemple, pour classifier des images on utiliserait `task="image-classification"`.
Le modèle par défaut est le transformeur visuel `vit` avec des sous-images de 16 et une taille initiale de 224x224 pixels.
On spécifie l'argument `model` ci-dessous pour clarifier le modèle utilisé.

```python
img_pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
```

```python
# Chat
preds = img_pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
preds
```

```python
# Mtl Bagel
img_pipeline("https://upload.wikimedia.org/wikipedia/commons/8/8c/Bagels-Montreal-REAL.jpg")
```

```python
# NY Bagel
img_pipeline("https://zenaskitchen.com/wp-content/uploads/2024/03/new-york-style-bagels.jpg")
```

```python
# Space shuttle
img_pipeline("https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcT8fYspmvzBIMbRW4eXMT65SVOej_SMq3WqTSXkc5uEvF9OX6XQBs5ilx2qzWZa8_VkwzG82u4C6_4KenH9wc8ZAmzNFri0WzRmdfHqY_w")
```

```python
# Galaxie
img_pipeline("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/NGC_4414_%28NASA-med%29.jpg/620px-NGC_4414_%28NASA-med%29.jpg")
```

```python
del img_pipeline
```

## Interface complète

Le `pipeline` ci-dessus nous permet de tester un modèle très rapidement, mais ne permet pas d'interagir avec un ensemble de données ou d'entraîner le modèle.

### Importation des données

Comme avec PyTorch, huggingface comprends plusieurs ensembles de données.
Pour y accéder, on peut utiliser la librairie [Datasets](https://huggingface.co/docs/datasets/index).

Ici, on importe seulement les 5000 premiers exemples des données d'entraînement pour réduire la taille des fichiers sur notre disque.
On pourra créer nos propre sous-ensembles à partir des données d'entraînement uniquement.

```python
from datasets import load_dataset

data = load_dataset("food101", split="train[:5000]")
data
```

On voit que les données contiennent des images et leurs annotation.
Séparons maintenant le tout avec 80% des exemples utilisés dans l'entraînement et le dernier 20% utilisé pour la validation.

```python
# TODO: Split train/test objects?
data = data.train_test_split(test_size=0.2)
data
```

Voyons voir de quoi a l'air un exemple:

```python
data["train"][0]
```

On voit que:

- Contrairement à PyTorch, qui nous donne des tuples, on a ici un dictionnaire.
- L'image est au format PIL, que nous avons vu plus tôt dans le cours
- L'annotation est un nombre entier, comme avec PyTorch

L'attribut `features` des données nous permet cependant d'accéder à un peu plus de détail sur les données

```python
test_num = 53
test_name = "steak"
labels = data["train"].features["label"]
print("Labels", labels)
print("Number of classes", labels.num_classes)
print(f"Class {test_num}", labels.int2str(test_num))
print(f"Class int for {test_name}", labels.str2int(test_name))
```

```python
label2id, id2label = dict(), dict()
for name in labels.names:
    i = labels.str2int(name)
    label2id[name] = i
    id2label[i] = name
```

```python
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng()

i = int(rng.integers(len(data["train"])))
eg = data["train"][i]
img, label = eg["image"], eg["label"]
print("Image format:", img.mode, img.size)

plt.imshow(img)
plt.title(id2label[label])
plt.show()
```

### Préparation des données

Comme avec PyTorch, il faut transformer les données de PIL vers des tenseurs.
Pour ce faire, Hugginface inclut des classes de type `Preprocessor`.
On peut utiliser le pre-processeur d'un modèle pré-entraîné, par exemple ViT entraîné sur les données ImageNet-21K.

```python
from transformers import AutoImageProcessor

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
image_processor
```

```python
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
torch_transforms = Compose([
    Resize(size),
    ToTensor(),
    normalize,
])
def transforms(examples):
    examples["pixel_values"] = [torch_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples
data = data.with_transform(transforms)
```

```python
data["train"][0]
```


```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
```


### Création du modèle

Plus haut, nous avons importé notre modèle via un `pipeline`.
Ici, nous allons plutôt importer le modèle directement.
Nous utiliserons tout de même un modèle pré-entraîné.

```python
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
)
```

L'avertissement ci-dessus nous indique que bien que le modèle soit pré-entraîné, son classificateur (la dernière couche) n'est pas entraîné.
Il faudra donc ajuster les poids et biais à la tâche qui nous intéresse ici.
Par contre, tout le reste du modèle est pré-entraîné.

Voyons voir de quoi est fait le modèle.

```python
model
```

Remarquez que la structure générale est la même que celle vue en clase:

- Un encodage des images et la position
- Un dropout optionnel
- Un encodeur, composé ici de 12 blocs ViT, qui eux contiennent:
  - Une couche d'attention
  - Une connection résiduelle
  - Une couche pleinement connectée
  - Des normalisations de couche (`LayerNorm`)
- Une classificateur permettant de convertir la sortie la sortie du classificateur en score pour chaque catégorie

Par défaut, le modèle contient uniquement deux sorties.
Il faut l'initialiser avec le bon nombre de classes.
On peut également utiliser la conversion entre les numéros de classes et leur nom.

```python
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels.names),
    id2label=id2label,
    label2id=label2id,
)
```

```python
model
```

### Entraînement

#### Métrique d'évaluation

https://huggingface.co/evaluate-metric
https://huggingface.co/docs/evaluate/index

```python
import evaluate

accuracy = evaluate.load("accuracy")
```

```python
accuracy
```

```python
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(predictions=preds, references=labels)
```


#### Boucle d'entraînement

```python
from transformers import TrainingArguments, Trainer
```

```python
training_args = TrainingArguments(
    output_dir="new_model",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
```

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)
```

```python
trainer.train()
```
