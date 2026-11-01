# Journal de veille — bouba_sens v0.7.1

Un bloc par tour. Chaque identifiant gardé a été résolu par API le jour du
tour (arXiv avec un User-Agent neutre et 3 s entre appels ; Crossref ;
DataCite ; Semantic Scholar). Aucun identifiant n'est repris de mémoire.

## Tour 1 — 2026-11-01

### Canaux interrogés

| canal | interrogé | outil | remarque |
|---|---|---|---|
| arXiv | oui | API export (https) | l'API en http renvoie 301 |
| OpenReview | non | — | reporté au tour 2 |
| bioRxiv | non | — | reporté au tour 2 |
| Crossref-journaux | oui | Crossref `/works/<doi>` + graphe S2 (références et citations) | OpenAlex a renvoyé 429 (quota journalier) sur les filtres de liste ; remplacé par S2 sans clé |
| thèses-HAL | non | — | reporté au tour 2 |
| code-données | oui | GitHub (dépôts des articles), OSF, Hugging Face | — |

### Requêtes exactes

arXiv (API export, `search_query=`) :

1. `all:bouba AND all:kiki` → 8 résultats.
2. `abs:"critical period" AND multimodal AND deprivation` → 0 résultat.
3. `abs:"sound symbolism" AND abs:audio` → 3 résultats.
4. `abs:"critical learning period"` → 10 résultats.

Remontée de citations S2, profondeur 1, sur 6 graines (`/paper/<id>/references`
et `/paper/<id>/citations`, 3 s entre appels) :

| graine | amont (références) | aval (citations) |
|---|---|---|
| arXiv:2210.04643 | 38 | 18 |
| arXiv:2310.16781 | 77 | 23 |
| DOI:10.1177/0956797619866625 | 36 | 26 |
| DOI:10.1038/s41598-023-38486-y | 58 | 8 |
| DOI:10.1098/rstb.2020.0390 | 113 | 177 |
| DOI:10.1093/brain/awt176 | 0 (S2 renvoie `data: null`, références masquées par l'éditeur) | 228 |

Code-données : dépôts `mjkleinman/CriticalPeriodMultiView` et
`TAU-VAILab/kiki-bouba` ; OSF 79gn4 (Sourav) et w7crs (Ćwiek) ; recherche HF
`bouba` / `kiki` sur les jeux de données.

### Comptes

- arXiv : 21 bruts → 20 dédoublonnés (une requête se recoupe) → 9 gardés
  (2310.16781, 2507.10013, 2407.17974, 2409.14917, 2409.12306, 2512.12245,
  2308.12221, 2506.15954, 2609.36475 ; les autres sont hors sujet ou
  seulement tangents, dont 2301.00267).
- Graphe S2 : 802 lignes brutes → 732 identifiants uniques
  (`snowball-tour1.tsv`, une ligne malformée retirée) → 22 gardés pour la
  note (liste ci-dessous).
- Code-données : 6 ressources vérifiées → 6 gardées.

### Entrées gardées (id résolu)

Lues en entier : arXiv:2210.04643, arXiv:2310.16781,
DOI:10.1177/0956797619866625, DOI:10.1038/s41598-023-38486-y,
DOI:10.1098/rstb.2020.0390, DOI:10.1093/brain/awt176.

[résumé moteur] :

- périodes critiques en réseau : arXiv:1711.08856, arXiv:2308.12221,
  arXiv:2312.00935, arXiv:2506.15954, arXiv:2608.25976 ;
- bouba/kiki et modèles : arXiv:2407.17974, arXiv:2507.10013,
  arXiv:2409.14917, arXiv:2409.12306, arXiv:2512.12245, arXiv:2607.10162,
  arXiv:2609.36475 ;
- humain, restauration et cécité : DOI:10.33735/phimisci.2024.11337 (cite
  Sourav et Piller), DOI:10.1016/j.cortex.2025.08.003 (cite Sourav, Piller et
  Ćwiek), DOI:10.1016/j.isci.2024.109167, DOI:10.53841/bpscog.2026.1.11.32,
  DOI:10.1016/j.cognition.2014.03.015, DOI:10.1016/j.cognition.2018.02.023,
  DOI:10.1016/j.cognition.2019.01.006, DOI:10.1016/j.neubiorev.2020.10.030,
  DOI:10.1111/desc.70099 ;
- méthode : arXiv:1603.05766 (p-valeurs de permutation jamais nulles) ;
- données : DOI:10.52202/068431-1003 (ActionSense), DOI:10.25592/uhhfdm.139
  (DataCite, données Sourav), DOI:10.17605/OSF.IO/Q6JYN (DataCite ; Crossref
  ne le connaît pas).

Code-données : `mjkleinman/CriticalPeriodMultiView` (Apache-2.0),
`TAU-VAILab/kiki-bouba` (MIT), OSF 79gn4, OSF w7crs,
HF `Rapidata/kiki-bouba-audio-20k` (aucune licence affichée),
HF `dlion168/bouba-kiki-lalm` (cc-by-4.0).

Échecs de résolution : aucun. Indisponibles : OpenAlex (429), texte de
Spence 2024 (page anti-robot Anubis, non contournée), PDF OUP de Collignon
(anti-robot ; version ORBi lue à la place).

### Ce que le tour a changé dans la note

- Kleinman Fig. 8 devient une antériorité directe du contraste précoce/tardif
  en réseau : la phrase de `docs/paper/main.tex:1100-1103` est fragilisée.
- Ćwiek 2022 donne 72 % et non ≥ 95 % : `docs/paper/main.tex:118` et
  `:1110-1111` sont à sourcer ou corriger.
- Sourav : SSA-V est au hasard chez CC et DC, et pas seulement « réduite »
  comme l'écrivait la revue de fond du 01/11.
- Arbitrage Sourav/Piller : la voie données (OSF) écarte le seul délai depuis
  l'opération comme explication ; écart à la règle du brief déclaré.
- Le test de couverture échoue (3/6 canaux) ; la saturation n'est pas
  atteinte.
