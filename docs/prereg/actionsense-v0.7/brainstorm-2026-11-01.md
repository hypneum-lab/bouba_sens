# Brainstorm — bouba_sens v0.7.1 (2026-11-01)

> **Aucun de ces dessins ne part avant que le verrou soit passé** (note de
> lecture, §6 : saturation, recoupement, couverture), ni avant qu'un
> amendement v0.7.1 soit signé par le PI et tagué. Ce document ne rédige pas
> l'amendement et ne pose aucun tag. Aucun calcul n'est lancé sur KXKM, macM1
> ou GrosMac ; tous les coûts sont estimés pour le CPU d'electron-server.

## Question ouverte rappelée

Une lésion tardive d'une seule modalité, qui atteint vraiment le plancher
−20 dB, laisse-t-elle une interaction lésion × requête asymétrique (Me6, B-3)
distincte d'une lésion congénitale, une fois retirés les effets principaux de
requête ?

Faits de départ :

- v0.7 ne passe jamais `--k-steps` ; avec k = 5 000 et 100 pas de lésion, le
  SNR atteint vaut 19,21 / 19,41 / 19,80 dB
  (`scripts/run_grid.sh:15,166-174` ;
  `docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md:34-36`).
- Le SNR atteint n'est pas consigné : `cli.py` enregistre `k_steps` nominal
  seulement.
- Le bruit est relatif à l'écart-type du lot et n'agit que pendant
  l'entraînement de phase 2 ; les sondes `query_successes` sont propres
  (`src/bouba_sens/loop.py:364-402`).
- ADR-0024 : seul le régime straight-through apprend.
- Le contrôle négatif 2 du brouillon #66 est tautologique : la moyenne sur
  réplicats de P_r − P_σ(r) est identiquement nulle, donc p ≡ 1.
- Sur le mock, le nul par blocs donne p = 1,0000 avec une médiane observée
  sous la médiane nulle
  (`docs/milestones/2026-10-31-me6-large-n-results.md:53`).

## Dessins

### D0 — Ne rien lancer ; requalifier v0.7 en non-test

- **Bras** : aucun.
- **Nul** : sans objet.
- **Critère de réfutation** : sans objet ; on écrit que v0.7 telle que taguée
  teste une lésion à environ 19-20 dB et ne peut pas réfuter une hypothèse sur
  −20 dB.
- **Coût** : 0 h machine ; environ 1 h humaine (une phrase dans l'amendement
  et dans le papier).
- **Ce que ça tranche** : rien sur la question ; évite seulement une
  conclusion fausse.
- **Antériorité la plus proche** : ADR-0020 (constat identique sur les grilles
  historiques).

### D1 — Lésion effective avec porte de manipulation en dB

- **Bras** : les 4 grilles v0.7 avec un calendrier qui atteint le plancher.
  Variante a : `k_steps = STEPS_LESION` (rampe jusqu'au dernier pas). Variante
  b : rampe sur la moitié des pas puis maintien au plancher. Graines neuves.
  Pilote sur les graines 100-101, qui sert déjà à la longueur de bloc.
- **Mesure** : à chaque pas, SNR appliqué = 10·log10(var signal / var bruit)
  sur les tenseurs réellement bruités ; consigné dans les métadonnées de la
  cellule.
- **Porte de manipulation** : pour chaque cellule floor, SNR mesuré au dernier
  pas dans [−21, −19] dB ; plancher tenu sur une fraction préenregistrée des
  pas (variante b). Contrôle positif : la diagonale de la modalité lésée chute
  plus en floor qu'en plus10 (dose-réponse monotone). Échec de la porte = la
  grille ne décide rien.
- **Porte d'apprentissage** : critère de perte de phase 1 préenregistré
  (ADR-0024), évalué sur le pilote.
- **Nul** : celui de la règle de décision retenue (D2 ou nul apparié
  ADR-0022) ; D1 ne change pas le nul, il change la manipulation.
- **Critère de réfutation** : porte de manipulation passée, porte
  d'apprentissage passée, et statistique de décision ≤ seuil B-3 avec TOST
  d'équivalence sur Δ → absence d'effet de lésion à −20 dB dans ce dessin.
- **Coût** : environ 155 tâches par grille × 85-112 s / PAR 4 ≈ 1 à 1,2 h CPU
  par grille, soit 4 à 5 h pour 4 grilles, plus environ 0,5 h de pilote
  (base : brouillon #66 §10 ; à réestimer pour ActionSense). Humain : environ
  4 h (instrumentation du SNR, porte, relecture).
- **Ce que ça tranche** : si une vraie lésion −20 dB modifie quelque chose ;
  sans D1, aucun résultat Me6/B-3 sur la lésion n'est interprétable.
- **Antériorité la plus proche** : flou 4× de Kleinman 2210.04643 (déficit
  fort et mesuré) ; ADR-0020 ; spike ADR-0019-b2 à −19,8 dB.

### D2 — H0 sur les résidus d'interaction

- **Bras** : aucun nouveau ; analyse des matrices P des grilles D1.
- **Nul** : ajustement additif ligne + colonne des 20 cellules hors diagonale
  par trio (effets principaux de requête et de modalité) ; matrice de résidus
  R à 11 degrés de liberté ; inversion de signe au niveau graine (ou par
  blocs, selon ADR-0023). p-valeur de permutation jamais nulle,
  (b + 1)/(B + 1) (arXiv:1603.05766).
- **Statistique** : médiane sur trios de max_{q≠m} |R[q,m] − R[m,q]|.
- **Critère de réfutation** : statistique ≤ 0,02 (seuil B-3, inégalité
  stricte pour passer) et TOST sur Δ préenregistrée → pas d'interaction.
- **Coût** : secondes de CPU ; humain environ 3 h (préreg du test, test
  unitaire sur données simulées avec et sans interaction).
- **Ce que ça tranche** : sépare l'interaction lésion × requête des effets
  principaux de requête construits par le monde (limite 4 d'ADR-0022).
- **Antériorité la plus proche** : préreg interne
  `docs/prereg/2026-10-26-b3-lesion-interaction.md:53-58` (même ajustement,
  déjà utilisée sur le monde gaussien).

### D3 — Contrôle 2 du brouillon #66 refait (moitiés disjointes)

- **Bras** : R ≥ 4 réplicats par (trio, modalité), comme #66 ; aucun bras
  propre si #66 n'est pas adopté.
- **Contrôle** : on coupe les réplicats en deux moitiés disjointes H1 et H2
  et on calcule A_neg = moyenne_H1(P[a,b] − P[b,a]) − moyenne_H2(P[a,b] −
  P[b,a]). Espérance nulle quelle que soit μ, non tautologique, et elle
  contient la variance inter-réplicat.
- **Nul** : distribution de A_neg sur des découpages aléatoires des
  réplicats ; contrôle de taille : taux de rejet à α dans α ± 2 erreurs-types.
- **Critère de réfutation** : si A_neg rejette plus souvent que α, le nul
  réseau est mal calibré et ne peut pas servir de règle de décision.
- **Coût** : celui de #66 : 755 tâches confirmatoires, 4,5 / 5,9 / 11 h selon
  la charge ; pilote 302 tâches, 1,8 / 2,3 / 4,4 h. Analyse : minutes.
  Humain : environ 3 h.
- **Ce que ça tranche** : si le nul réseau proposé par #66 a la bonne taille.
- **Antériorité la plus proche** : brouillon #66 (contrôles lignes 182-206) ;
  relecture croisée du 01/11 (REVISE).

### D4 — Calibration du nul par blocs sous prédicteur constant (M25)

- **Bras** : simulation numpy seule, sans réseau.
- **Générateur** : prédicteur quasi constant (audio et vision d'accord sur
  81-100 % des items,
  `docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md:39-42`), parts
  de classes 0,43 / 0,12 / 0,09 / 0,37
  (`docs/osf/amendment-v0.7-actionsense.md:543`), étiquettes en segments
  contigus, longueurs de bloc b ∈ {1, b_monde, 2·b_monde}.
- **Nul** : le nul par blocs d'ADR-0021/0022 appliqué tel quel à des données
  sans effet par construction.
- **Critère de réfutation** : taux de rejet à α hors de α ± 2 erreurs-types,
  ou p-valeurs non uniformes (test KS) → le nul par blocs est mal calibré
  dans ce régime, et une grille qui retombe dans ce régime ne décide pas.
- **Coût** : quelques minutes CPU ; humain environ 3 h.
- **Ce que ça tranche** : si le p = 1,0000 du mock vient d'un nul
  sur-dispersé ou d'un vrai effet nul.
- **Antériorité la plus proche** : ADR-0021, ADR-0022 ; M25 du rapport du
  01/11.

### D5 — Bras de restauration type Kleinman (axe délai depuis la restauration)

- **Bras** : lésion précoce levée après N pas, puis évaluation à plusieurs
  délais, pour imiter l'axe « délai depuis l'opération » de Sourav/Piller.
- **Nul** : même nul que D2.
- **Critère de réfutation** : pas de dépendance au délai au-delà du nul.
- **Coût** : au moins 3 fois D1 (plusieurs délais) ; humain environ 8 h.
- **Ce que ça tranche** : une analogie computationnelle avec la divergence
  Sourav/Piller, sans lecture bouba/kiki directe dans ActionSense.
- **Antériorité la plus proche** : Kleinman 2210.04643 (déficit temporaire
  puis récupération, Fig. 8) ; largement devancé.

## Classement « ce que ça tranche / coût »

| rang | dessin | ce que ça tranche | coût machine | rapport |
|---|---|---|---|---|
| 1 | D4 | calibration du nul en régime dégénéré | minutes | très élevé |
| 2 | D2 | interaction isolée des effets principaux | secondes | très élevé (dépend de données D1) |
| 3 | D1 | rend la lésion réelle et mesurée | 4-5 h | élevé ; prérequis de toute lecture |
| 4 | D3 | taille du nul réseau | 4,5-11 h | moyen ; seulement si #66 est adopté |
| 5 | D0 | évite une conclusion fausse | 0 | utile mais ne tranche rien |
| 6 | D5 | analogie Sourav/Piller | > 12 h | faible ; devancé par Kleinman |

Inférence : la séquence la moins chère qui tranche est D4 puis D2 (outils
validés sur simulation), puis D1 avec D2 en règle de décision.

## Ce qu'on ne fait pas, et pourquoi

- **D5 maintenant** : devancé par Kleinman et hors périmètre d'ActionSense
  (pas de lecture bouba/kiki) ; reporté après M3.
- **Relancer v0.7 telle quelle** : la lésion n'atteint pas le plancher ; le
  résultat serait un non-test.
- **Ajouter des corrections de multiplicité à v0.7 a posteriori** : changerait
  une règle taguée ; c'est à décider dans v0.7.1, avant données.
- **Calcul sur KXKM, macM1 ou GrosMac** : interdit par le brief ; tout est
  dimensionné pour electron-server.

## Éléments que devra contenir l'amendement v0.7.1 (liste, non rédigée)

1. Calendrier de lésion qui atteint le plancher (`K_STEPS` ou rampe +
   maintien), déclaré comme écart par rapport aux grilles historiques.
2. SNR réellement appliqué, mesuré et consigné dans les métadonnées de chaque
   cellule.
3. Porte de manipulation en dB (tolérance, fraction des pas, contrôle positif
   dose-réponse) et conséquence d'un échec.
4. Porte d'apprentissage préenregistrée (critère de phase 1, ADR-0024).
5. Alignement du texte sur l'inégalité stricte « > »
   (`docs/osf/amendment-v0.7-actionsense.md:437` contre `:408`).
6. H0 sur les résidus d'interaction (D2), primaire ou secondaire, à côté de
   la règle figée, avec marge Δ du TOST.
7. Calibration du nul par blocs sous prédicteur constant (D4) faite et
   rapportée avant le tag.
8. Contrôle 2 de #66 refait (D3) si le nul réseau est adopté.
9. Règle de multiplicité pour les 12 tests (correction ou justification
   explicite de son absence).
10. Hôte electron-server et présence vérifiée des tenseurs ActionSense.
11. Qualification par ADR-0024 (seul straight-through apprend) : statut des
    grilles argmax-bridge.
12. Antériorités citées : Kleinman 2210.04643 (Fig. 8), Ćwiek 2022 (72 %).
13. Graines neuves, distinctes de 300-304 (non utilisées à ce jour).
14. Nouveau tag de gel après signature du PI.
