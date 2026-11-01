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

## Tour 2 — 2026-11-01

Les six canaux sont interrogés à ce tour. Les requêtes sont écrites telles
que lancées. Agent de requête : `hypneum-lab reading-gate` ; aucune adresse
e-mail du PI n'a été envoyée (le paramètre `email` d'Unpaywall, obligatoire,
a reçu une adresse publique générique du service, `unpaywall@impactstory.org`).

### Canaux interrogés

| canal | interrogé | outil | remarque |
|---|---|---|---|
| arXiv | contrôle seulement | API export, 3,5 s entre appels | 7 requêtes de confirmation de l'absence (ci-dessous) ; l'inventaire reste celui du tour 1 |
| OpenReview | oui | API `api2.openreview.net/notes/search` | recherche plein texte ; `notes?id=`, `notes?content.venueid=` et les PDF répondent 403 (challenge) : « non vérifié » pour ces accès |
| bioRxiv | oui | API `details` (catégorie neuroscience) + Europe PMC | voir « Écarts » : page de 30 et non de 100, balayage toutes catégories abandonné |
| Crossref-journaux | oui | Crossref `/works/<doi>`, Unpaywall, S2 (snowball) | Spence 2024 et Vreugdenhil 2025 ouverts en PDF |
| thèses-HAL | oui | API HAL (`docType_s:THESE`), API theses.fr | premières réponses seulement (15 par requête HAL ; 8 à 30 par requête theses.fr) |
| code-données | oui | `gh search repos`, API Hugging Face, Zenodo, API OSF v2, clone du dépôt Kleinman | |

### Requêtes exactes

OpenReview (`https://api2.openreview.net/notes/search?term=<terme>&limit=40`, avec `&group=<groupe>` quand indiqué) :

1. Sans groupe (5) : `bouba kiki` ; `sound symbolism` ; `critical period multisensory` ; `critical learning period multimodal` ; `sensory deprivation multimodal network`.
2. Groupes `ICLR.cc/2027/Conference`, `ICLR.cc/2026/Conference`, `NeurIPS.cc/2025/Conference`, `ICLR.cc/2025/Conference` (7 termes chacun, 28 requêtes) : `critical learning period` ; `critical period` ; `bouba kiki` ; `sound symbolism` ; `congenital blindness cataract` ; `sensory deprivation` ; `cross-modal plasticity`.
3. Complémentaires (4) : `multimodal lesion modality dropout robustness early late` (ICLR 2027) ; `critical period multisensory integration deep networks` (ICLR 2026, NeurIPS 2025, ICLR 2025).

bioRxiv :

- API : `https://api.biorxiv.org/details/biorxiv/<d1>/<d2>/<curseur>?category=neuroscience`, du 2024-01-01 au 2026-11-01, huit tranches lancées en parallèle (semaines de 7 jours), curseur par pas de la taille de page renvoyée (30). Filtre local sur titre + résumé : `bouba|kiki|sound[- ]symbolism|sound[- ]shape|crossmodal correspondence|cross-modal correspondence|critical period|critical learning period|sensitive period|sight restoration|cataract|congenital(ly)? blind|early[- ]blind|late[- ]blind|visual deprivation|sensory deprivation` (insensible à la casse).
- Premier balayage toutes catégories, même filtre, **abandonné** (trop lent : 6 à 9 semaines sur environ 19 par tranche) ; ses 172 lignes de résultats ne sont pas utilisées sauf pour confirmer que le filtre ramène du bruit.
- Europe PMC (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`, `query=` ci-dessous, `resultType=lite`, `pageSize=100`), chaque requête étant suivie de `AND (SRC:PPR AND PUBLISHER:"bioRxiv") AND FIRST_PDATE:[2024-01-01 TO 2026-11-01]` :
  1. `TITLE_ABS:"bouba" OR TITLE_ABS:"kiki"`
  2. `TITLE_ABS:"sound symbolism" OR TITLE_ABS:"sound-shape"`
  3. `TITLE_ABS:"critical period" AND TITLE_ABS:"deep neural network*"`
  4. `TITLE_ABS:"congenital" AND TITLE_ABS:"cataract" AND TITLE_ABS:"sight restoration"`
  5. `TITLE_ABS:"late blind" AND TITLE_ABS:"congenitally blind" AND TITLE_ABS:"crossmodal"`
  6. `(TITLE_ABS:"congenital*" OR TITLE_ABS:"early blind" OR TITLE_ABS:"early-blind") AND TITLE_ABS:"blind*" AND (TITLE_ABS:"sound" OR TITLE_ABS:"audio*" OR TITLE_ABS:"crossmodal" OR TITLE_ABS:"cross-modal")`
  7. `TITLE_ABS:"sight restoration" OR TITLE_ABS:"cataract-reversal" OR TITLE_ABS:"Project Prakash"`
  8. `TITLE_ABS:"sensitive period*" AND (TITLE_ABS:"multisensory" OR TITLE_ABS:"crossmodal" OR TITLE_ABS:"cross-modal")`
  9. `TITLE_ABS:"critical period*" AND (TITLE_ABS:"neural network*" OR TITLE_ABS:"artificial" OR TITLE_ABS:"deep learning" OR TITLE_ABS:"model*") AND TITLE_ABS:"visual deprivation"`
  10. `TITLE_ABS:"visual deprivation" AND (TITLE_ABS:"neural network*" OR TITLE_ABS:"deep learning" OR TITLE_ABS:"computational model*")`
  11. `TITLE_ABS:"shape-sound" OR TITLE_ABS:"phonetic" AND TITLE_ABS:"blind*" AND TITLE_ABS:"iconic*"`

Thèses-HAL :

- HAL : `https://api.archives-ouvertes.fr/search/?q=<terme>&fq=docType_s:THESE&fl=title_s,uri_s,authFullName_s,producedDateY_i&rows=15` pour `bouba kiki` ; `symbolisme sonore forme` ; `période critique multisensoriel` ; `cataracte congénitale` ; `cécité congénitale intermodal` ; `sound symbolism` ; `critical period deep neural network` ; `privation sensorielle précoce réseau de neurones` ; `sensitive period multisensory`.
- theses.fr : `https://theses.fr/api/v1/theses/recherche/?q=<terme>&nombre=<n>` pour `bouba kiki` (10) ; `symbolisme sonore` (10) ; `cécité congénitale` (30) ; `période critique` (10) ; `plasticité intermodale` (10) ; `symbolisme sonore aveugle` (8) ; `privation visuelle précoce` (8) ; `réseau de neurones période critique développement` (8) ; `cross-modal correspondences blind` (8).

Code-données :

- GitHub : `gh search repos "<terme>" --sort updated --limit 20 --json fullName,pushedAt,stargazersCount,license,description` pour `bouba kiki` ; `critical period multisensory` ; `critical learning period deep network` ; `sound symbolism` ; `sensory deprivation neural network` ; `congenital blindness model` ; puis (`--limit 12`) `critical learning periods` ; `critical period deep learning` ; `critical-period neural network` ; `multisensory critical period` ; `modality deprivation multimodal`.
- Hugging Face : `https://huggingface.co/api/{models,datasets}?search=<terme>&limit=20&sort=lastModified&direction=-1` pour `bouba` ; `kiki` ; `sound symbolism` ; `critical period` ; `sensory deprivation` ; `cataract`.
- Zenodo : `https://zenodo.org/api/records?q=<terme>&sort=mostrecent&size=12` pour `bouba kiki` ; `"sound symbolism" AND blind` ; `"critical period" AND "deep" AND multisensory` ; `congenital cataract AND "sound-shape"` ; `sight restoration AND crossmodal`.
- OSF : `https://api.osf.io/v2/registrations/?filter[title][icontains]=<terme>&page[size]=15&sort=-date_created` pour `bouba` ; `sound symbolism` ; `critical period` ; `sound-shape` ; `congenital cataract` ; `sight restoration`.
- Dépôt de code : clone de `mjkleinman/CriticalPeriodMultiView` (commit 3e86c6c).

Contrôle arXiv (API export, 3,5 s entre appels) : `ti:"Critical Learning Periods" AND ti:Revisiting` (0) ; `abs:"critical learning period" AND abs:"modality"` (1, arXiv:2210.04643) ; `abs:"critical period" AND abs:"multimodal" AND abs:"early"` (1, hors sujet) ; `abs:"sensory deprivation" AND abs:"neural network"` (1, hors sujet) ; `abs:"cataract" AND abs:"neural network" AND abs:"visual experience"` (0) ; `abs:"congenital" AND abs:"visual deprivation" AND abs:"deep"` (0) ; `abs:"newly sighted" OR abs:"sight restoration" AND abs:"deep"` (1, hors sujet). Cinq recherches par titre ont servi en plus à résoudre des identifiants (Hong et Ibaraki, Jeong, Lou et al.).

Snowball S2 (`scripts/citation_snowball.py --depth 1`), 5 graines absentes du tour 1 : DOI:10.33735/phimisci.2024.11337 (Spence 2024), DOI:10.1016/j.cortex.2025.08.003 (Vreugdenhil 2025), arXiv:2308.12221 (Kleinman 2023), arXiv:2506.15954 (Fukase 2025), arXiv:2507.10013 (Kouwenhoven 2025). Le script plafonne la profondeur à 1.

### Comptes

| canal | brut | dédoublonné | gardé | détail |
|---|---|---|---|---|
| OpenReview | 1 295 lignes (37 requêtes ; la première passe avait donné 945 identifiants dédoublonnés, la relance 1 118 : la recherche n'est pas déterministe) | 1 118 | 5 | 12 travaux repérés par le titre ; 1 nouveau (Jeong, AAAI-26), 3 versions évaluées d'entrées du tour 1 (Kouwenhoven, Tseng, Kleinman), 1 doublon du tour 1 (Hong et Ibaraki, soumission ICLR 2027 = arXiv:2609.36475). Non gardés : Lou et Liu (réseaux à une modalité), « Computational measurement of perceived pointiness from pronunciation » (revue MTA 2024, hors sujet) et quatre doublons du tour 1 (Alper, Verhoef, Loakman, Sharma) |
| bioRxiv, API neuroscience | 38 126 lignes de versions, 1 344 appels | 33 529 DOI → 231 DOI avec un mot-clé du filtre | 6 | lecture des titres des 231 ; les 6 sont du contexte humain (restauration de la vue, déprivation précoce, bouba/kiki dans le cortex visuel), sans effet sur une phrase |
| bioRxiv, Europe PMC | 24 | 19 | 2 | chicks naïfs (bouba/kiki) et macaques (sound-shape), contexte non humain |
| Thèses-HAL | HAL 25 + theses.fr 69 = 94 | HAL 22, theses.fr 68 (les deux se recoupent) | 1 | De Carolis 2019 (symbolisme sonore) ; aucune thèse sur déprivation précoce + association son-forme |
| Code-données | GitHub 45, Hugging Face 95, Zenodo 17, OSF 37 | environ 190 | 8 | trois dépôts, un jeu HF, quatre enregistrements OSF |
| Snowball S2 | 319 lignes | 293 (240 absents du tour 1) | 7 directs + 18 contexte | `snowball-tour2.tsv` (brut) et `snowball-tour2.annot.tsv` (annoté) |

### Entrées gardées (id résolu par `scripts/resolve_ids.py` ou par API ce jour)

Avec effet sur la note :

| id résolu | auteurs, année | titre | ce que ça change | statut de lecture |
|---|---|---|---|---|
| arXiv:2507.10013 (v2) ; OpenReview gkcU26BOml | Kouwenhoven, Shahrasbi, Verhoef, 2025 | Cross-modal Associations in Vision and Language Models: Revisiting the Bouba-Kiki Effect | **infirme** la ligne 4 sous sa forme « les VLM montrent l'effet » : CLIP ResNet et ViT, aucun effet cohérent | corps du texte lu (p. 1-10) |
| arXiv:2511.10045 (v6) | Jeong et al., 2025 (AAAI-26) | Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism | **dépasse** la ligne 5 : MLLM, texte / IPA / audio, 8 052 mots ; outil : jeu LEX-ICON, dépôt `jjhsnail0822/sound-symbolism` | résumé et introduction |
| arXiv:2409.12306 ; DOI:10.1109/SLT61566.2024.10832254 | Tseng et al., 2024 | Measuring Sound Symbolism in Audio-Visual Models | **nuance** la ligne 5 : AUC phonétique 0,87 (SpeechCLIP) à 0,31 (CAV-MAE) ; version SLT 2024 = second témoin | résumé, tableau 2 |
| arXiv:2607.10162 (v2) | Tsai et al., 2026 | Hearing Like Humans? Sound Symbolism and Perceptual Alignment in Speech Language Models | **nuance** la ligne 5 : les SLM alignent mal | résumé, introduction |
| arXiv:2407.17974 | Verhoef, Shahrasbi, Kouwenhoven, 2024 | What does Kiki look like? | **nuance** la ligne 4 ; source du « 95 % » (Ramachandran et Hubbard 2001) pour la ligne 3 | résumé, introduction |
| DOI:10.1109/CVPR52729.2023.02327 | Kleinman, Achille, Soatto, 2023 | Critical Learning Periods for Multisensory Integration in Deep Networks | **corrige** « ICLR 2023 » en CVPR 2023 ; confirme l'existence de la version évaluée | métadonnées, README du dépôt |
| arXiv:2506.15954 | Fukase et al., 2025 | One Period to Rule Them All: Identifying Critical Learning Periods in Deep Networks | **infirme** son statut de réplication indépendante du résultat multisensoriel (ligne 1) | résumé, introduction, mots-clés |
| DOI:10.1117/12.3094028 | Jakowski, Santos-Villalobos, 2026 | Semantic similarity deficits in neural network critical learning periods | **confirme** (hors groupe, une modalité) le contraste précoce/tardif, ligne 2a | [résumé moteur] (S2) |
| DOI:10.33735/phimisci.2024.11337 | Spence, Di Stefano, 2024 | Old and new versions of the Molyneux question | **change** la ligne 7 / §3.1 : les tiers lisent Sourav et Piller comme convergents | résumé, note 7, références |
| DOI:10.1016/j.cortex.2025.08.003 | Vreugdenhil et al., 2025 | Exploring touch-colour associations in achromatopsia: A case study | **idem** | introduction, discussion, références |
| DOI:10.17605/OSF.IO/SQ2PY (+ DOI:10.17605/OSF.IO/VNZSU, DOI:10.17605/OSF.IO/4W8PD) | Sourav et al., 2019 | A Protracted Sensitive Period… (registre figé : projet, données, matériel) | **confirme** « non préenregistrée » (second témoin) | métadonnées et résumé de registre |
| `haeliotang/critical-period-lm` | dépôt individuel, 2026 | critical-period-lm | **outil et contrepoint** (ligne 2a) : essai auto-déclaré préenregistré, LM, verdict `REVERSE_ONSET_EFFECT` | début du README |
| `mjkleinman/CriticalPeriodMultiView` (commit 3e86c6c) | Kleinman et al. | dépôt de code | **change** la ligne 2b : pas de code pour la Fig. 8 (vidéo) | lu (README, scripts, `models.py`) |
| `Rapidata/psychology-association-kiki-bouba-etc` | Rapidata, 2025-2026 | jeu HF | **outil** (réponses humaines par paire d'images, sans licence affichée) | métadonnées |
| 7 entrées directes du snowball : DOI:10.1117/12.3094028, DOI:10.1177/0956797614563957, DOI:10.1016/j.cub.2015.06.040, S2:886b8eda…, S2:4d72336d…, S2:d8a74bd5…, S2:8535f2d6… | Jakowski 2026 ; Huber et al. 2015 ; McKyton et al. 2015 ; Iida et Funakura 2024 ; Cuskley et al. 2018 ; (« Magnitude symbolism without sound ») ; Deroy et al. 2013 | voir `snowball-tour2.annot.tsv` | **candidates à lire** ; aucune ne modifie encore une phrase | titre et résumé S2 |

Retenues comme contexte, sans effet sur une phrase de la note (titre lu seulement, [titre seul]) :

- DOI:10.1101/2025.03.25.645249 (Sourav et al., laboratoire Röder, restauration de la vue après cataracte congénitale ; non indépendant du groupe Sourav), DOI:10.1101/2025.09.09.675092 (Hassett et al., substance blanche après restauration), DOI:10.1101/2024.11.22.624459 (Rączy et al., représentations visuelles et auditives après restauration), DOI:10.1101/2024.11.30.625697 (Mattioni et al., privation visuelle néonatale transitoire), DOI:10.64898/2026.06.14.732169 (Wei et al., intégration sensorielle après déprivation précoce), DOI:10.64898/2026.01.17.700088 (Ioannucci et al., bouba/kiki dans le cortex visuel précoce).
- DOI:10.1101/2024.05.17.594640 (Loconsole et al., poussins naïfs) et DOI:10.64898/2026.07.31.741992 (Loconsole et al., macaques) : effet son-forme chez des animaux sans langage. Inférence : à distinguer de l'effet humain dans toute citation.
- DOI:10.70675/2ab5d1fcz4b0ez4723zb4fbz2500278f4585 (De Carolis 2019, thèse Lyon 2 de symbolisme sonore et protocoles de tâches d'association).
- OSF `s8prt` (2026-08-13, enregistrement d'une étude de nourrissons de 18 à 20 mois sur l'effet bouba/kiki) : développement de l'effet, en cours.

Non gardés mais vus : Lou et al. (ICLR 2025, soumission retirée, « Revisiting Critical Learning Periods in Deep Neural Networks »), Liu et Gurevych (ICLR 2025, rejetée, début d'entraînement et généralisation hors distribution) : réseaux à une modalité, sans lésion de modalité.

Échecs de résolution : aucun. Indisponibles : PDF et `notes?id=` d'OpenReview (403 ; la soumission ICLR 2027 de Hong et Ibaraki, id `2FaHRono5T`, est « non vérifiée » côté OpenReview, sa version arXiv est arXiv:2609.36475), SPIE (Jakowski, accès fermé), PDF OUP (Brain, déjà au tour 1), Ramachandran et Hubbard 2001 (non ouvert).

### Écarts à la consigne (déclarés)

- bioRxiv : la page de l'API renvoie 30 enregistrements et non 100 ; le balayage toutes catégories sur 2024-2026 demandait plus de 4 000 appels et a été remplacé par la seule catégorie neuroscience (1 344 appels, achevé), complété par Europe PMC. Une étude bouba/kiki animale ou en psychologie classée dans une autre catégorie peut avoir échappé au balayage de l'API ; les requêtes Europe PMC (toutes catégories) la rattrapent en partie.
- OpenReview : `notes?content.venueid=…` (requête de la consigne) répond 403 ; remplacé par `notes/search` (plein texte, avec `group=`).
- Ne pas refaire le comptage OSF de Sourav : respecté. Aucune source publiée lue ne le rapporte (voir la note, §3.1).
- Lecture : plusieurs PDF ouverts n'ont été lus que par passages (marqués dans la note, §2) ; aucun n'est compté « lu en entier ».

### Snowball

Sources de départ : les cinq ci-dessus. Fichiers : `snowball-tour2.tsv` (sortie brute du script : direction, id, année, titre, premier auteur, source), `snowball-tour2.annot.tsv` (avec deux colonnes ajoutées par lecture des titres : `nouveau_vs_tour1`, `pertinence_titre` ∈ direct / contexte / hors). 319 lignes, 293 identifiants uniques, 240 absents de `snowball-tour1.tsv`, dont 7 directs et 18 de contexte. Le bruit vient surtout des références amont (219 amont contre 22 aval parmi les 241 nouvelles avant retrait du doublon S2 d'Achille ; psychophysique et philosophie anciennes, apprentissage automatique générique).

### Ce que le tour a changé dans la note

- §2 : CVPR 2023 et non ICLR 2023 ; description exacte de la Fig. 8 (vidéo, Multi-View Transformer) ; ajout de Kouwenhoven 2025 (corps du texte lu) ; réorganisation des lectures partielles.
- §3 ligne 1 : Fukase n'est pas un second article indépendant. Ligne 2 scindée en 2a (tient) et 2b (**à recouper**, le code ne reproduit pas la Fig. 8). Ligne 3 : origine du « 95 % ». Ligne 4 : devient « contesté ». Ligne 5 : devient « hétérogène » et perd sa marque. Ligne 7 et §3.1 : les tiers lisent Sourav et Piller comme convergents ; Sourav non préenregistrée confirmé par OSF ; aucune source publiée ne rapporte le comptage des délais.
- §4 ligne 1 : preuve de l'absence étendue à six canaux, marque « à recouper » levée sous réserves.
- §6 : trois tests mis à jour. §7 : journal complété.

### Saturation

Entrées nouvelles pertinentes ce tour : 7 directes sur 240 nouvelles du snowball (2,9 %) ; 25 si on compte le contexte (10,4 %). Hors snowball : 1 travail nouveau et directement pertinent (Jeong, AAAI-26) sur 12 repérés par OpenReview. Le critère « < 5 % » est atteint sur la lecture étroite, non sur la lecture large ; la définition de « pertinent » reste au contradicteur. Profondeur 2 non faite (le script la plafonne).
