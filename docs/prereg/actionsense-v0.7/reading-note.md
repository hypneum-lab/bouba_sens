# Note de lecture — bouba_sens v0.7.1 (v0, 2026-11-01)

> Statut : v0, tour de veille 1. Ce document ne rend aucun verdict sur
> E1-small, N10-C run 2 ou Me6. Il ne modifie ni l'amendement v0.7
> (`docs/osf/amendment-v0.7-actionsense.md`), ni aucune préreg, ni aucun
> ADR. Aucun tag n'est posé. Le contradicteur doit encore la relire.

## 1. Question que la campagne prétend trancher (une phrase)

Une lésion sensorielle tardive (T2), appliquée à une seule modalité d'un
réseau multimodal entraîné sur des capteurs réels (ActionSense), laisse-t-elle
une trace asymétrique dans les réponses croisées (Me6, B-3) qui la distingue
d'une lésion congénitale (T1), au-delà de ce que produisent les seuls effets
principaux de requête ?

Fait : la grille v0.7 lance 4 grilles de 150 cellules sur les graines 300-304
avec `STEPS_LESION=100` par défaut (`scripts/run_grid.sh:15`) et ne passe
jamais `--k-steps` (`scripts/run_grid.sh:166-174`).
Fait : avec k = 5 000 par défaut, le SNR atteint au dernier pas de lésion vaut
19,21 / 19,41 / 19,80 dB pour les cellules floor / minus10 / plus10
(`docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md:34-36`).
Inférence : telle que taguée, v0.7 teste une lésion faible (bruit d'environ
11 % de l'écart-type du lot), pas la lésion à −20 dB que décrit la question.
C'est le constat B16 du rapport du 01/11 (`synthese/rapport.md:67`).

## 2. Sources primaires lues en entier

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| arXiv:2210.04643 (v2) | Kleinman, Achille, Soatto, 2022 (ICLR 2023) | 1-17, annexes comprises | Un déficit temporaire précoce (flou 4×, ou décorrélation des vues), suivi d'un entraînement normal de 180 époques, dégrade durablement l'intégration multisensorielle d'un réseau profond. Fig. 8 : le même déficit, introduit à des moments variables (fenêtre glissante), donne un contraste précoce/tardif, avec jusqu'à environ 20 % de perte pour un déficit précoce et une perte quasi nulle pour un déficit tardif. La reconstruction croisée supprime la période critique. Les sources y sont symétriques ; l'asymétrie entre modalités est laissée à des travaux futurs. Code `mjkleinman/CriticalPeriodMultiView` (Apache-2.0). |
| arXiv:2310.16781 (v3) | Alper, Averbuch-Elor, 2023 (NeurIPS 2023) | 1-23, annexes comprises | Sondage zéro coup de CLIP et Stable Diffusion avec des pseudo-mots (CV)₁(CV)₂(CV)₁ : AUC 0,74 (CLIP) et 0,77 (SD), ΔP_kb 80 % et 52 %. Étude utilisateur : 73 % et 55 % (89 % et 78 % en modèle mixte). Les auteurs ne tranchent pas entre graphie et son. Le concept est quasi absent de LAION (12 légendes). Code `TAU-VAILab/kiki-bouba` (MIT). |
| DOI:10.1177/0956797619866625 | Sourav et al., 2019 (Psychol. Sci.) | texte intégral PMC6787766, toutes sections et tableaux | Groupes CC 30, DC 24, CB 15, LB 12, TS 70. Association son-forme haptique (SSA-H) au hasard chez CC (β 1,230, p = .165), DC (p = .782) et CB (p = .247) ; intacte chez LB (β 3,287, p = .042). Version visuelle (SSA-V) au hasard chez CC (p = .578) et DC (p = .416), réduite par rapport aux TS. CC opérés en moyenne à 58 mois, testés en moyenne à 18,9 ans. Étude non préenregistrée. Données ouvertes OSF 79gn4 (composant evpc2, DOI 10.25592/uhhfdm.139). |
| DOI:10.1038/s41598-023-38486-y | Piller et al., 2023 (Sci. Rep.) | texte intégral PMC10349879, toutes sections et figures | 23 CC opérés en Éthiopie, âgés de 5 à 18 ans, opérés en moyenne à 9,8 ans. Essai visuel unique. Taux bouba/kiki 0,74 contre 0,93 chez les contrôles. Moins de 6 mois après l'opération : 0,58 (p = 0,57) ; plus de 6 mois : 0,91 (p = 0,03), indistinguable des contrôles (p = 0,81). La coupure à 6 mois est choisie après coup. Les auteurs attribuent l'écart avec Sourav aux stimuli (formes noires pleines, protrusions plus profondes). Données « sur demande raisonnable ». |
| DOI:10.1098/rstb.2020.0390 | Ćwiek et al., 2022 (Phil. Trans. R. Soc. B 377:20200390, en ligne 2021-11-15) | texte intégral PMC8591387 | 917 participants, 25 langues, 10 systèmes d'écriture. Accord global 72 % [IC 56-82]. 17/25 langues au-dessus du hasard ; bouba plus robuste que kiki (22/25 contre 11/25). Effet de l'écriture faible. Données et scripts ouverts (OSF w7crs). |
| DOI:10.1093/brain/awt176 | Collignon et al., 2013 (Brain) | 1-15, version Advance Access déposée sur ORBi (ULiège) | CB 12, LB 10 (cécité acquise en moyenne à 36 ans). Recrutement du cuneus et spécialisation spatiale dorsale droite chez les seuls CB. DCM : couplage direct A1→V1 chez CB, rétroaction via l'IPS chez LB. Chez LB, corrélation négative avec la durée de cécité. La plasticité tardive existe mais n'est pas spécifique. |

Hors tableau, lues en résumé seulement (« [résumé moteur] ») : Spence 2024
(DOI:10.33735/phimisci.2024.11337, page anti-robot Anubis), Senna 2024
(DOI:10.1016/j.isci.2024.109167), Senna 2026 (DOI:10.53841/bpscog.2026.1.11.32),
Vreugdenhil 2025 (DOI:10.1016/j.cortex.2025.08.003), Fryer 2014
(DOI:10.1016/j.cognition.2014.03.015), Hamilton-Fletcher 2018
(DOI:10.1016/j.cognition.2018.02.023), Bottini 2019
(DOI:10.1016/j.cognition.2019.01.006), Vogelsang 2025 (DOI:10.1111/desc.70099),
Achille 2019 (arXiv:1711.08856), Kleinman 2023 (arXiv:2308.12221), Zhang 2023
(arXiv:2312.00935), Fukase 2025 (arXiv:2506.15954), Plantinga 2026
(arXiv:2608.25976), Röder 2020 (DOI:10.1016/j.neubiorev.2020.10.030), Tseng
2024 (arXiv:2409.12306), Verhoef 2024 (arXiv:2407.17974), Kouwenhoven 2025
(arXiv:2507.10013), Tsai 2026 (arXiv:2607.10162), Phipson & Smyth
(arXiv:1603.05766).

## 3. Déjà publié

Chaque ligne donne au moins deux sources de nature différente, sinon elle porte
« à recouper ».

1. **Les réseaux profonds ont des périodes critiques d'intégration
   multisensorielle.** Article arXiv:2210.04643 (lu en entier) ; dépôt de code
   `mjkleinman/CriticalPeriodMultiView` (existence et licence vérifiées) ;
   article indépendant arXiv:2506.15954 (Fukase 2025, [résumé moteur]).
   Réserve : 1711.08856 et 2308.12221 viennent du même groupe.
2. **Un contraste « déficit précoce contre déficit tardif » existe déjà dans un
   réseau artificiel.** Article arXiv:2210.04643, Fig. 8 (fenêtre glissante).
   À recouper : le code n'a pas été lu ligne à ligne pour confirmer qu'il
   produit la Fig. 8. Fait : `docs/paper/main.tex:1100-1103` écrit « We are not
   aware of an artificial-network model that directly contrasts congenital and
   late sensory loss ». Inférence : cette phrase est fragilisée par la Fig. 8
   de Kleinman, même si le déficit y est symétrique et non une lésion d'une
   seule modalité.
3. **L'accord inter-culturel bouba/kiki est d'environ 72 %, pas ≥ 95 %.**
   Article DOI:10.1098/rstb.2020.0390 (lu en entier) ; données OSF w7crs
   (dossiers data, scripts, audio vérifiés, non réanalysés). Fait :
   `docs/paper/main.tex:118` et `:1110-1111` annoncent « ≥ 95 % cross-cultural
   agreement ». Inférence : ce chiffre doit être sourcé ou corrigé.
4. **Les modèles vision-langage montrent un effet bouba/kiki en zéro coup.**
   Article arXiv:2310.16781 (lu en entier) ; code `TAU-VAILab/kiki-bouba` ;
   article arXiv:2407.17974 (Verhoef 2024, [résumé moteur]). Nuance :
   arXiv:2507.10013 (Kouwenhoven 2025, [résumé moteur]) rapporte des résultats
   moins nets selon les modèles.
5. **Une association son-forme mesurée sur modèles audio existe.** arXiv:2409.12306
   (Tseng 2024) et arXiv:2607.10162 (Tsai 2026), tous deux [résumé moteur] ;
   jeux HF `Rapidata/kiki-bouba-audio-20k` (sans licence affichée) et
   `dlion168/bouba-kiki-lalm` (cc-by-4.0). À recouper (aucun texte lu en entier).
6. **Chez l'humain, la cécité tardive laisse une plasticité réelle mais non
   spécifique.** Article DOI:10.1093/brain/awt176 (lu en entier, IRMf + DCM) ;
   article DOI:10.1177/0956797619866625 (comportemental : LB intacts en SSA-H,
   lu en entier). Natures différentes (neuro-imagerie et comportement), même
   réseau de co-auteurs élargi (Röder/Collignon) : indépendance partielle.
7. **Une période sensible pour l'association son-forme chez les enfants nés
   aveugles puis opérés.** Contesté : Sourav 2019 dit oui, Piller 2023 dit non.
   Voir l'arbitrage 3.1. Pas de consensus publié.

### 3.1 Arbitrage Sourav 2019 / Piller 2023 (deux voies de recoupement)

Faits communs : les deux études portent sur des CC opérés de cataracte
bilatérale dense. Sourav conclut à une période sensible prolongée ; Piller
« réfute » une période sensible précoce.

**Voie 1 — données ouvertes de Sourav.** Fait : le fichier OSF
`Sourav_et_al_2019_PSCI_SSA_Data_Randomized.xlsx` (sha256 86c7c631…aea1)
randomise par groupe les colonnes âge, acuité et âge à l'opération (clé A,
partagée entre CC et DC), découplées des réponses. Fait : les paires âge au
test / âge à l'opération semblent conservées entre elles (aucun des 30 délais
calculés n'est négatif). Fait : 29 CC sur 30 ont été testés plus de 6 mois
après l'opération (délais de 4 à 540 mois). Inférence : le régime « plus de
6 mois » où Piller retrouve l'effet ne suffit pas à expliquer l'absence d'effet
chez Sourav, puisque presque tous ses CC y sont. Les CC de Sourav ont aussi été
opérés plus tôt (58 mois contre 9,8 ans). Restent comme explications
candidates : l'âge au test (adultes contre enfants), les stimuli, la
population, l'essai unique de Piller et sa coupure choisie après coup.
Limite : la randomisation empêche de relier délai et réponse individuelle. Il
faut demander les données non randomisées au laboratoire Röder et celles de
Piller (« sur demande raisonnable »).

**Écart déclaré à la règle du brief.** Ce comptage descriptif sur un fichier
public dépasse l'usage « scripts seulement pour résoudre des identifiants ». Il
est signalé ici et doit être refait par le contradicteur avant d'être repris.

**Voie 2 — tiers et revues.** Spence 2024 (DOI:10.33735/phimisci.2024.11337),
revue de la question de Molyneux, cite les deux études ([résumé moteur], page
anti-robot non contournée). Vreugdenhil 2025 (DOI:10.1016/j.cortex.2025.08.003)
est la seule autre œuvre du graphe S2 qui cite les deux ([résumé moteur]).
Senna 2024 et Senna 2026 viennent du groupe de Piller : non indépendants. La
littérature convergente sur les aveugles congénitaux va plutôt dans le sens
d'un rôle de l'expérience visuelle (Fryer 2014, Hamilton-Fletcher 2018), avec
un résultat contraire (Bottini 2019) ; ces trois-là sont [résumé moteur].

Conclusion de l'arbitrage, à ce tour : la question « période sensible ou pas »
reste ouverte côté humain. Inférence : bouba_sens ne peut s'appuyer sur aucune
des deux études comme vérité de terrain ; il peut seulement citer la
divergence. À recouper au tour 2 (lecture intégrale de Spence 2024).

## 4. Reste ouvert

1. **Lésion asymétrique d'une seule modalité, congénitale contre tardive, sur
   capteurs réels, avec lecture croisée de type Me6/B-3.** Preuve de l'absence
   à ce tour : arXiv `abs:"critical period" AND multimodal AND deprivation`
   = 0 résultat ; `abs:"critical learning period"` = 10 résultats, aucun avec
   lésion d'une seule modalité ; les 18 œuvres qui citent arXiv:2210.04643
   (S2) ne la font pas, d'après les titres et résumés ; Kleinman laisse
   l'asymétrie à des travaux futurs. Limite : OpenReview, bioRxiv et HAL n'ont
   pas été interrogés. À recouper : la couverture échoue à ce tour (§6).
2. **Dans bouba_sens même : l'effet d'une lésion qui atteint vraiment
   −20 dB.** Preuve : aucune grille n'a jamais tourné avec un calendrier qui
   atteint le plancher (`scripts/run_grid.sh:15,166-174` ;
   `docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md:34-36`).
   C'est un trou interne, pas un trou de littérature.
3. **Un H0 qui isole l'interaction lésion × requête.** Fait : la règle de
   symétrie Me6 confond interaction et effets principaux de requête
   (limite 4 d'ADR-0022, corrigée le 26/10). Fait : une préreg interne
   propose déjà un ajustement additif des 20 cellules hors diagonale et un
   test d'inversion de signe sur les résidus
   (`docs/prereg/2026-10-26-b3-lesion-interaction.md:53-58`). Inférence :
   l'outil existe dans le dépôt, mais v0.7 ne l'utilise pas en règle de
   décision.
4. **Un nul par blocs calibré quand le prédicteur est quasi constant.** Fait :
   sur le mock, médiane observée 0,0088 contre médiane nulle 0,0223 et P99
   0,0313, p = 1,0000 (`docs/milestones/2026-10-31-me6-large-n-results.md:53`).
   Fait : les vecteurs de requête audio et vision s'accordent sur 81-100 % des
   items (`docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md:39-42`).
   Inférence : le nul par blocs paraît sur-dispersé dans ce régime ; sa taille
   réelle n'a pas été mesurée.

## 5. Dessin minimal qui tranche la question ouverte

Combinaison des dessins D1 et D2 du brainstorm (`brainstorm-2026-11-01.md`).

- **Bras.** Les 4 grilles v0.7 inchangées (nolock/lock × argmax-bridge/
  straight-through, 5 graines × 5 modalités × T1/T2 × 3 SNR), avec un
  calendrier qui atteint le plancher : `k_steps = STEPS_LESION` (rampe
  linéaire jusqu'au plancher au dernier pas) ou rampe sur la moitié des pas
  puis maintien. Graines neuves, distinctes de 300-304 (non utilisées à ce
  jour, mais déjà associées à la v0.7 taguée).
- **Porte de manipulation.** Le SNR réellement appliqué est mesuré à chaque
  pas, 10·log10(var signal / var bruit) sur les tenseurs bruités, et consigné.
  Elle passe si, pour chaque cellule floor, le SNR mesuré au dernier pas est à
  ±1 dB du nominal (−20 dB) et si le plancher est tenu sur une fraction
  préenregistrée des pas. Contrôle positif : la diagonale de la modalité lésée
  chute davantage en floor qu'en plus10.
- **Nul.** H0 « pas d'interaction lésion × requête » : ajustement additif des
  20 cellules hors diagonale par trio, résidus R, statistique = médiane sur
  trios de max |R[q,m] − R[m,q]|, inversion de signe au niveau graine.
- **Critère.** Seuil B-3 (> 0,02, inégalité stricte) sur la statistique des
  résidus, p au niveau graine sous le nul ; absence déclarée seulement par
  TOST sur une marge Δ préenregistrée.
- **Coût.** Estimation à partir de #66 §10 (85 à 112 s par tâche, PAR 4) :
  environ 1 à 1,2 h CPU par grille sur electron-server, plus le pilote
  100-101. À vérifier : présence des tenseurs ActionSense sur electron-server.

Aucun de ces bras ne part avant que le verrou soit passé et qu'un amendement
v0.7.1 soit signé par le PI et tagué.

## 6. Trois tests du verrou

| test | état à ce tour | ce qui manque |
|---|---|---|
| Saturation | Échoue. Tour 1 seulement ; 732 entrées dédoublonnées dans `snowball-tour1.tsv`, triées sur titres. Kleinman Fig. 8 est une antériorité trouvée tard, ce qui indique que la recherche n'est pas saturée. | Tour 2 : profondeur 2 depuis Kleinman, Spence 2024 et Vreugdenhil 2025 ; mesurer la part de nouvelles entrées pertinentes. |
| Recoupement | Partiel. Lignes 1, 3, 4 et 6 de §3 tiennent par deux natures. Lignes 2, 5 et l'arbitrage 3.1 portent « à recouper ». | Code Kleinman lu pour la Fig. 8 ; Spence 2024 lu en entier ; comptage OSF refait par le contradicteur ; demande des données non randomisées. |
| Couverture (6 canaux) | Échoue : 3/6 canaux interrogés (arXiv, Crossref-journaux via S2, code-données). | OpenReview, bioRxiv, thèses-HAL. OpenAlex à reprendre (429 le 01/11). |

## 7. Journal de lecture

| date | source | pages |
|---|---|---|
| 2026-11-01 | arXiv:2210.04643 v2, PDF arXiv | 1-17 |
| 2026-11-01 | arXiv:2310.16781 v3, PDF arXiv | 1-23 |
| 2026-11-01 | DOI:10.1177/0956797619866625, PMC6787766 (texte intégral via Unpaywall) | toutes sections, tableaux |
| 2026-11-01 | OSF 79gn4/evpc2, fichier xlsx des données randomisées | feuille complète |
| 2026-11-01 | DOI:10.1038/s41598-023-38486-y, PMC10349879 | toutes sections, figures |
| 2026-11-01 | DOI:10.1098/rstb.2020.0390, PMC8591387 | toutes sections |
| 2026-11-01 | DOI:10.1093/brain/awt176, ORBi (Advance Access, accès libre vert ; PDF OUP bloqué par anti-robot) | 1-15 |
| 2026-11-01 | DOI:10.33735/phimisci.2024.11337 | résumé seulement (anti-robot) |
| 2026-11-01 | Dépôt : amendement v0.7, ADR-0020, 0022, 0024, 0027, brouillon #66, `run_grid.sh`, `lesion.py`, `cli.py`, `loop.py` | lus en entier ou aux lignes citées |

Non lus en détail à ce tour : ADR-0019 (deux fichiers), 0021, 0023, 0026.
