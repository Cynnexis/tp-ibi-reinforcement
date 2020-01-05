# TP IBI Renforcement

![language: python][shield-language]

> Présenté par
> * Anthony BACCUET
> * Valentin BERGER

## Getting Started

Ces étapes vous permettront de lancer le projet depuis votre machine. La première étape consiste à cloner le projet :

```bash
git clone https://github.com/Cynnexis/tp-ibi-reinforcement.git
```

... et se rendre dans le dossier cloné :

```bash
cd tp-ibi-reinforcement
```

### Prérequis

Les scripts utilisent des notations de Python 3. Par la suite, nous estimerons que la commande `python` lancera Python version 3.

Le projet peut s'exécuter en utilisant pip, venv ou conda (voir partie suivante) ou avec Docker (voir partie **Installation (docker)**)

> ⚠️ Recommandation:
>
> Il est recommandé de suivre l'installation via pip, venv ou conda, car les scripts Python du projet utilise `matplotlib`, une librairie non supportée sous Docker sans GUI.

### Installation (pip/venv/conda🐍)

Pour installer les modules Python nécessaire à l'exécution du script, il faut utiliser le fichier `requirements.txt`, qui contient l'ensemble des dépendances Python.

> A noter qu'il est conseillé d'utiliser un environnement virtuel tel que venv ou conda pour installer les modules.
>
> Sous venv :
>
> ```bash
> python -m venv .
> ```
> 
> Puis activer l'environnement avec la commande suivante sous UNIX:
>
> ```bash
> source venv/bin/activate
> ```
> 
> sous Windows:
>
> ```bash
> venv/Scripts/activate.bat
> ```
> 
> En utilisant conda :
>
> ```bash
> conda create --name tp-ibi-reinforcement-baccuet-berger
> activate tp-ibi-reinforcement-baccuet-berger
> ```

Pour installer les dépendances, utilisez la commande suivante :

Avec Pip :

```bash
pip install -r requirements.txt
```

Avec conda :

```bash
conda install -n tp-ibi-reinforcement-baccuet-berger --yes --file requirements.txt
```

### Installation (docker🐳)

Cette partie vous permettera d'exécuter le script Python depuis un container Docker. Ce type d'installation n'est pas recommandé car les diagrammes `matplotlib` ouvert par le script ne pourront pas être visible.

1. Lancez l'application Docker.
2. Depuis un terminal ouvert à la racine du projet, lancez :
```bash
docker-compose up
```
3. Le script s'exécute dans la console.
4. Pour stopper le script, pensez à faire :
```
docker-compose down
```

## Built With

Le projet fut développé avec :

* [Python 3][python3]
* [PyTorch][pytorch]
* [PyCharm][pycharm]
* [Virtualenv][venv]
* [Docker][docker]

## Versioning

Ce projet est versioné avec Git/GitHub.

Lien GitHub: https://github.com/Cynnexis/tp-ibi-reinforcement

## Authors

* Anthony BACCUET
* Valentin BERGER

[shield-language]: https://img.shields.io/badge/language-python-yellow.svg
[python3]: https://www.python.org/download/releases/3.0/
[pytorch]: https://pytorch.org/
[pycharm]: https://www.jetbrains.com/pycharm/
[venv]: https://virtualenv.pypa.io
[docker]: https://www.docker.com/