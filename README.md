# Deep Decoding

Implémentation du deep-decoder en MATLAB et [Julia](https://julialang.org/).

## Installation

### Julia

L'installation de Julia se fait sur leur [site officiel](https://julialang.org/downloads/).

### Packages

Pour installer les packages nécessaires, il suffit de lancer le fichier `install.jl` avec la commande suivante:

```bash

julia install.jl

```

(Il manque peut être le paquet random je sais plus s'il est fourni de base ou pas mais si il manque il suffit de l'installer avec `Pkg.add("Random") dans le fichier install.jl`)


## Utilisation

Pour lancer le programme, il suffit de lancer le fichier `deep-decoder-inpainting.jl` avec la commande suivante:

```bash
julia deep-decoder-inpainting.jl
```

Le programme est un peu lourd parce qu'il a besoin d'un nombre important de paramètres (k dans le fichier .jl défini en haut)

Les résultats sont sauvegardés dans le répertoire courant sous le nom `deep-decoder.png`.

Insh ca marche lol
