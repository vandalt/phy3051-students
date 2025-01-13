# Tutoriel Git pour PHY3051

Ce document contient un résumé des commandes Git vues en classe.

Les commandes utilisées ci-dessous sont toutes disponibles sur Linux et MacOS (Unix) et (je crois) sur Windows PowerShell. Pour toutes les commandes ci-dessous, ajouter `--help` devrait fournir de l'information utile.

### Ouvrir un terminal

Pour ce tutoriel, nous allons surtout travailler dans le terminal. Le terminal donne une interface de commande textuelle ("ligne de commande", ou CLI pour _Commande line interface_ en anglais) qui permet de manipuler des fichiers et d'exécuter des programmes.

Vous pouvez ouvrir l'application "Terminal" de votre ordinateur. Pour Windows, je suggère d'utiliser le "Anaconda Prompt" s'il vous utilisez Anaconda, ou bien Windows PowerShell.

Le terminal s'ouvre généralement dans le répertoire (synonyme de "dossier") _home_, dénoté par `~`. Dans _home_, on trouve généralement des dossiers "Documents", "Downloads", etc., mais vous êtes aussi libres d'ajouter d'autres dossier.

Je suggère de créer un dossier dans lequel vous entreposez vos codes, par exemple:

```bash
mkdir ~/codes
```

et de vous y déplacer avec

```bash
cd ~/codes
```

### Créer un répertoire Git

On peut créer un répertoire et y entrer, puis s'assurer qu'il est bien vide

```bash
mkdir demo-tp1
cd demo-tp1
ls -a
```

La commande `ls`  liste les fichiers et dossiers dans le répertoire.
L'argument `-a` signifie "_all_" et permet de lister tous les fichiers, incluant les fichiers "cachés", commençant par un point.
Dans ce cas-ci, vous devriez voir uniquement `. ..` dans la liste. `.` dénote le répertoire actuel et `..` le répertoire parent. `cd ..` permet de se déplacer dans le parent (qui ici devrait être `~/codes`).

Pour que Git puisse versionner notre répertoire, il faut l'initialiser. Git sait ainsi qu'il faut suivre les changements.

```bash
git init
```

Une fois que c'est fait, on peut voir que le dossier `.git` a été ajouté au répertoire. C'est un dossier caché (débute par un point), donc on ne le verra pas avec `ls`, mais seulement avec `ls -a`. Vous n'aurez presque jamais à interagir directement avec ce dossier. Pour voir le statut du répertoire, on peut utiliser

```bash
git status
```

### Créer un fichier

On peut ensuite créer un premier fichier. Il est typique de documenter un projet avec un fichier nommé `README.md`. Le suffixe `md` signifie _Markdown_. C'est un format texte qui permet d'avoir des sections en italique ou en gras, des blocs de code, des listes et des section (c'est d'ailleurs le format utilisé pour rédiger ce guide). La plupart des éditeurs de textes, ainsi que GitHub, reconnaissent ce type de fichier automatiquement.

On peut créer un fichier directement dans le terminal (`touch README.md` sur Unix et `type nul > README.md` sur Windows, je crois). Sinon, on peut simplement l'ouvrir avec un éditeur de texte et le sauvegarder sans rien écrire. Une fois que c'est fait, `ls` devrait montrer le ficher. Maintenant, `git status` devrait afficher le ficher `README.md` comme étant _untracked_: on sait que le fichier est là, mais Git ne fait rien avec pour l'instant.

Pour dire à Git de suivre un fichier, il faut d'abord l'ajouter avec

```bash
git add README.md
```

Maintenant, Git va suivre l'état du fichier, mais de l'a pas encore enregistré dans l'historique. Pour ce faire, il faut utiliser un _commit_:

```bash
git commit -m "First commit"
```

Ici, l'argument `-m` permet d'inclure le message directement. Sinon, une fenêtre s'ouvrirait pour qu'on puisse écrire notre message. Une fois que c'est fait `git status` devrait indiquer que tout est "clean".

Pour voir l'historique des _commits_, on peut utiliser

```bash
git log
```

### Modifier un fichier

Une fois notre fichier README créé, on peut y ajouter du texte. Avec un éditeur de texte, ouvrez le fichier avec un éditeur de texte (VSCode, Notepad, Sublime Text, Vim, Nano, alouette), puis ajoutez y le titre du projet ainsi qu'une courte description. Par exemple:

```markdown
<!-- Ceci est un commentaire markdown et ne sera pas visible sur GitHub -->
<!-- "#" Indique qu'il s'agit d'un titre. -->
# TP 1 pour PHY3051
Premier TP PHY3051: Introduction à Git et révision Python.

Voici du texte _italique_ et **gras**.

Voici un bloc de code Python utilisant `numpy`:

<!-- Enlever les barres obliques -->
\`\`\`python
import numpy as np
\`\`\`
```

Une fois ces modifications sauvegardées, `git status` devrait signaler que des fichiers ont été modifiés.
Pour voir ces différences plus en détail, on peut utiliser `git diff`.
Pour ajouter les modifications à l'historique de notre répertoire, il suffit de répéter les commandes  `add` et `commit`:

```bash
# "git add ." fonctionnerait aussi, ajoutant TOUS les fichiers et dossiers
git add README.md
git commit -m "Add text to README"
```

### Synchronisation avec GitHub

[GitHub](https://github.com/) est une plateforme qui permet de publier du code en ligne dans un "dépôt" (_repository_).
Un dépôt GitHub peut être public ou privé. Une fois le dépôt GitHub créé on peut y envoyer du code à partir de notre terminal avec Git.
Pour ce faire, vous aurez besoin:

- D'un compte GitHub
- D'une authentification GitHub sur votre ordinateur (je suggère d'utiliser une clé SSH: voir [ce guide](https://docs.github.com/en/get-started/quickstart/set-up-git#authenticating-with-github-from-git) et [celui-ci](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account))

Une fois que c'est fait, on peut créer notre dépôt GitHub:

1. Rendez-vous sur GitHub et cliquez sur "New" ou "New repository" dans le menu.
2. Nommez le dépôt comme votre dossier local (facultatif mais aide à s'y retrouver).
3. Il y a ensuite plusieurs options, notamment de rendre le dépôt public ou privé, d'ajouter un fichier README ou un fichier .gitignore (pour ignorer certains fichiers dans le dépôt).
Pour l'instant, on peut laisser les options par défaut.
4. Créez le dépôt en confirmant le tout.
5. Copiez les commandes dans le menu. Ceci permet de dire à Git (sur votre ordinateur) où se trouve le dépot en ligne.

Voilà, votre dépôt est créé! Cependant, il est vide.
Pour envoyer le contenu de notre ordinateur vers GitHub, il faut utiliser la commande "push":

```bash
git push -u origin main
```

La partie `-u origin main` est requise uniquement pour la première synchronisation. Ensuite `git push` suffira.
Vous pouvez maintenant rafraîchir la page en ligne et vous devriez voir votre README s'afficher.

### Gitignore

Par défaut, Git considère tous les fichiers dans le répertoire.
Le fichier `.gitignore` permet de dire à Git quels fichiers **ne pas suivre**.

Voici un template utile pour Python: <https://github.com/github/gitignore/blob/main/Python.gitignore>.

Vous pouvez aussi manuellement y ajouter des fichiers, dossiers, ou _patterns_ (par exemple `data/*.csv` pour ignorer tous les fichiers CSV dans le dossier `data`).

### Autres interfaces pour Git

Le terminal est la principale interface pour Git, mais il en existe plein d'autres.
J'ai compilé une liste [ici](https://github.com/vandalt/phy3051-6051-ressources/#git).
Sinon, les IDEs comme VSCode et PyCharm on souvent leur propre interface pour Git.
