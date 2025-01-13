# Environnements virtuels pour Python

Pour fonctionner, Python a besoin d'un _environnement_: un ensemble de dossiers et fichiers installés sur votre ordinateur permettant d'exécuter des programmes Python et d'entreposer des _packages_.

Si vous installez Python directement sur votre système (avec votre distribution Linux ou à partir des instructions sur le site de Python), il s'agit d'un premier environnement Python.

Si vous installez Python avec [Anaconda](https://www.anaconda.com/download/) ou [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Python sera probablement installé dans votre "home" (`~`) sous `~/anaconda3` ou `~/miniconda3`. Il s'agit, encore une fois, d'un environnement.

**Plusieurs environnements Python peuvent coexister sur un même ordinateur**. C'est d'un côté très pratique, mais ça peut aussi être une [source de confusion](https://xkcd.com/1987/)...

Par défaut, Anaconda et Miniconda créent un environnement de base, nommé `base`. Il s'agit de l'installation initiale de Python. **Il est recommandé de ne pas utiliser `base` directement pour travailler sur vos projets** (ni l'installation Python "system", d'ailleurs). À la place, il est préférable de créer plusieurs environnements virtuels.

Ceci dit, il est techniquement possible de travailler à partir de base ou sur votre installation Python "system". Il est simplement probable que des problèmes surviennent un jour ou l'autre avec des conflits entre les dépendances de différents projets.

Dans ce tutoriel, nous couvrirons le concept des _environnements virtuels_. Les environnements virtuels permettent, sans requérir d'installer Python au complet chaque fois, d'avoir des environnements séparés pour différents projets ou différentes sphères de recherches.

<!-- TODO: Add link to last section -->
Il existe plusieurs façon de créer et gérer des environnements virtuels. Ici, j'utiliserai `conda`, qui est selon moi l'option la plus simple. Il existe cependant des alternatives, que je mentionne à la toute fin du document.

Dans les prochaines sections, nous allons couvrir:

<!-- TODO: Add links to sections -->
- L'installation Python avec `conda`
- La création d'environnements virtuels
- L'installation de _packages_ Python dans un environnement virtuel
- L'utilisation d'environnements virtuels avec des Jupyter Notebooks

## Installer Python

Si vous n'avez pas déjà `conda` sur votre système, vous pouvez suivre les instructions pour l'installer aux liens suivants:

- Anaconda: <https://www.anaconda.com/download/>
- Miniconda: <https://docs.conda.io/projects/miniconda/en/latest/>

Je suggère Miniconda, qui n'installe que le minimum requis et prend moins d'espace.

Une fois l'installation terminée, vous devriez avoir un dossier `miniconda3` dans votre "home". Vous pouvez le vérifier avec la commande `ls` (ou `dir` sur Windows). Sur Linux et MacOS, vous devriez aussi voir "(base)" affiché dans votre terminal. Sur Windows, le programme "Anaconda Prompt" ou "Anaconda PowerShell" devrait être disponible. Ouvrez le et "(base)" devrait également y être affiiché.

## Créer un environnement virtuel

Une fois que tout ceci est fait, rendez-vous dans le dossier que nous avons créé lors du [tutoriel Git](tp1-demo-git-python/00_tutoriel_git.md).

Pour créer un environnement avec `conda`, on peut utiliser la commande:

```bash
conda create -n phy3051 python pip ipython ipykernel
```

Ainsi, on crée un environnement et on s'assure que Python, Pip et IPython sont installés. Vous pouvez remplacer `phy3051` par le nom que vous souhaitez. Pour installer une version spécifique de Python, par exemple la version 3.11, remplacez `python` par `python=3.11`. Par défaut, `conda` installe la plus récente disponible.

Pour vous assurer que l'environnement a été créé, vous pouvez faire une liste de vos environnements avec

```bash
conda env list
```

Vous pouvez ensuite **activer l'environnement** avec

```bash
conda activate phy3051
```

Activer l'environnement signifie que la version de Python utilisée en tapant `python` ou `ipython` dans le terminal sera celle associée à l'environnement virtuel.

### Petite note sur les Jupyter Notebooks

Pour fonctionner avec un environnement virtuel, Jupyter doit savoir que cet environnement existe.

Si vous n'avez pas d'installation globale de Jupyter Notebooks ou JupyterLab, vous pouvez également ajouter `jupyterlab` à la fin de la commande pour l'installer dans l'environnement.

Si vous avez une installation globale, exécutez simplement la commande

```bash
python -m ipykernel install --user --name=phy3051
```

(en remplaçant `phy3051` par le nom de votre environnement). Le _kernel_ de l'environnement sera ensuite disponible pour JupyterLab.

## Installation de _packages_ Python

En plus de gérer les environnements virtuels, `conda` permet d'installer des packages Python. Or, ce ne sont pas tous les packages qui sont distribués via `conda`; beaucoup sont seulement accessibles via Pip. Pour cette raison, une fois l'environnement créé, je suggère d'utiliser Pip pour installer **tous les _packages_**.

Par défaut, notre environnement virtuel n'inclut aucun _package_. Pour en installer un, on peut utiliser

```bash
python -m pip install -U numpy
```

Ici, `-U` signifie _ugrade_ et permet de mettre à jour `numpy` s'il est déjà installé.

Si vous souhaitez garder une liste des _packages_ installés dans votre environnement, je suggère d'utiliser un fichier `requirements.txt`. Il s'agit simplement d'une liste de _pakages_ (un par ligne). Par exemple, nous pouvons créer un fichier avec le contenu suivant:

```
numpy
scipy
matplotlib
```

Il s'agit des _packages_ requis pour le TP d'aujourd'hui. Une fois le fichier créé, on peut installer les _packages_ avec

```bash
python -m pip install -U -r requirements.txt
```

Voilà! Vous pouvez maintenant ouvrir le Jupyter Notebook du TP d'aujourd'hui: [02_notebook_revision_python.ipynb](tp1-demo-git-python/02_revision_python.ipynb)

## Alternatives

Si vous préférez ne pas utiliser `conda`, des alternatives existent:

- [venv](https://docs.python.org/3/library/venv.html) - Pour créer des environnements virutels avec Python sans Conda
- [virtualenv](https://virtualenv.pypa.io/en/latest/)- Comme `venv`, mais avec un peu plus de fonctionnalités
- [pyenv](https://github.com/pyenv/pyenv) - Alternative à Conda pour utiliser différentes version de Python (bug un peu plus avec certains _packages_ scientifiques dans  mon expérience)
