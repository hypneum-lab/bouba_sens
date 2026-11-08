# Note de lecture — bouba_sens v0.7.1 (v0.2, 2026-11-01)

> Statut : v0.2, tours de veille 1 et 2 (`veille-journal.md`). Ce document ne
> rend aucun verdict sur E1-small, N10-C run 2 ou Me6. Il ne modifie ni
> l'amendement v0.7 (`docs/osf/amendment-v0.7-actionsense.md`), ni aucune
> préreg, ni aucun ADR. Aucun tag n'est posé. Le contradicteur doit encore la
> relire, et lui seul refait le comptage OSF de la voie 1 (§3.1). Seul le PI
> déclare le verrou passé ou non ; cette note ne le déclare pas.

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
| arXiv:2210.04643 (v2) | Kleinman, Achille, Soatto, 2022 (CVPR 2023, « Highlighted Paper » ; DOI:10.1109/CVPR52729.2023.02327) | 1-17, annexes comprises | Un déficit temporaire précoce (flou 4×, ou décorrélation des vues), suivi d'un entraînement normal de 180 époques, dégrade durablement l'intégration multisensorielle d'un réseau profond (expériences CIFAR-10, ResNet à deux vues). Fig. 8 (§4.3-4.4, Multi-View Transformer sur vidéo, classification d'actions) : la décorrélation des deux images, introduite dans une fenêtre glissante de taille fixe à des moments variables, donne un contraste précoce/tardif ; jusqu'à environ 20 % de perte de précision de test pour l'objectif supervisé quand le déficit est précoce (trace bleue). L'objectif de masquage avec reconstruction croisée n'a pas de période critique (trace rouge). Les sources y sont symétriques ; l'asymétrie entre modalités est laissée à des travaux futurs. Code `mjkleinman/CriticalPeriodMultiView` (Apache-2.0, dernier commit 2023-12-17) : voir §3, ligne 2. Correction du tour 2 : la note v0 écrivait « ICLR 2023 » ; le commentaire arXiv et le README du dépôt disent CVPR 2023. |
| arXiv:2507.10013 (v2) | Kouwenhoven, Shahrasbi, Verhoef, 2025 (NeurIPS 2025, poster) | 1-10 (corps du texte) ; annexes non lues | Deux variantes de CLIP (ResNet-50, ViT) sondées par probabilités d'étiquettes et par Grad-CAM, avec dix prompts et quatre sources de pseudo-mots. Les adjectifs anglais servent de base : appariés à environ 80 %. Les pseudo-mots ne sont pas appariés de façon fiable ; aucun des deux modèles n'atteint un niveau humain sur bouba/kiki avec Grad-CAM (performance pas fiablement au-dessus du hasard à 25 %). Les auteurs contredisent Alper et Averbuch-Elor (2023), jugent que leur méthode d'embeddings peut capter des régularités propres à une langue (Iida et Funakura 2024, VLM japonais, sans effet robuste) et notent que Tseng et al. (2024) utilisent la même méthode. Limites : seulement CLIP ; stimuli présentés en texte. Données et code OSF gqsv6 (cité, non ouvert). |
| arXiv:2310.16781 (v3) | Alper, Averbuch-Elor, 2023 (NeurIPS 2023) | 1-23, annexes comprises | Sondage zéro coup de CLIP et Stable Diffusion avec des pseudo-mots (CV)₁(CV)₂(CV)₁ : AUC 0,74 (CLIP) et 0,77 (SD), ΔP_kb 80 % et 52 %. Étude utilisateur : 73 % et 55 % (89 % et 78 % en modèle mixte). Les auteurs ne tranchent pas entre graphie et son. Le concept est quasi absent de LAION (12 légendes). Code `TAU-VAILab/kiki-bouba` (MIT). |
| DOI:10.1177/0956797619866625 | Sourav et al., 2019 (Psychol. Sci.) | texte intégral PMC6787766, toutes sections et tableaux | Groupes CC 30, DC 24, CB 15, LB 12, TS 70. Association son-forme haptique (SSA-H) au hasard chez CC (β 1,230, p = .165), DC (p = .782) et CB (p = .247) ; intacte chez LB (β 3,287, p = .042). Version visuelle (SSA-V) au hasard chez CC (p = .578) et DC (p = .416), réduite par rapport aux TS. CC opérés en moyenne à 58 mois, testés en moyenne à 18,9 ans. Étude non préenregistrée. Données ouvertes OSF 79gn4 (composant evpc2, DOI 10.25592/uhhfdm.139). |
| DOI:10.1038/s41598-023-38486-y | Piller et al., 2023 (Sci. Rep.) | texte intégral PMC10349879, toutes sections et figures | 23 CC opérés en Éthiopie, âgés de 5 à 18 ans, opérés en moyenne à 9,8 ans. Essai visuel unique. Taux bouba/kiki 0,74 contre 0,93 chez les contrôles. Moins de 6 mois après l'opération : 0,58 (p = 0,57) ; plus de 6 mois : 0,91 (p = 0,03), indistinguable des contrôles (p = 0,81). La coupure à 6 mois est choisie après coup. Les auteurs attribuent l'écart avec Sourav aux stimuli (formes noires pleines, protrusions plus profondes). Données « sur demande raisonnable ». |
| DOI:10.1098/rstb.2020.0390 | Ćwiek et al., 2022 (Phil. Trans. R. Soc. B 377:20200390, en ligne 2021-11-15) | texte intégral PMC8591387 | 917 participants, 25 langues, 10 systèmes d'écriture. Accord global 72 % [IC 56-82]. 17/25 langues au-dessus du hasard ; bouba plus robuste que kiki (22/25 contre 11/25). Effet de l'écriture faible. Données et scripts ouverts (OSF w7crs). |
| DOI:10.1093/brain/awt176 | Collignon et al., 2013 (Brain) | 1-15, version Advance Access déposée sur ORBi (ULiège) | CB 12, LB 10 (cécité acquise en moyenne à 36 ans). Recrutement du cuneus et spécialisation spatiale dorsale droite chez les seuls CB. DCM : couplage direct A1→V1 chez CB, rétroaction via l'IPS chez LB. Chez LB, corrélation négative avec la durée de cécité. La plasticité tardive existe mais n'est pas spécifique. |

Hors tableau, PDF ouvert au tour 2 mais **passages seulement** (« PDF,
passages ») : Spence et Di Stefano 2024 (DOI:10.33735/phimisci.2024.11337 ; la
page de l'éditeur s'est ouverte au tour 2 sans écran anti-robot ; lus : résumé,
note 7, références ; recherche par mots-clés, aucune autre mention de
Sourav/Piller/bouba), Vreugdenhil 2025 (DOI:10.1016/j.cortex.2025.08.003,
version déposée UvA ; lus : introduction, discussion et références aux lignes
citant Sourav/Piller/Ćwiek), Fukase 2025 (arXiv:2506.15954 ; lus : résumé,
introduction, recherche des mentions multimodales : aucune expérience
multimodale), Tseng 2024 (arXiv:2409.12306 ; lus : résumé, §5.1-5.2 avec le
tableau 2), Verhoef 2024 (arXiv:2407.17974 ; lus : résumé, introduction),
Tsai 2026 (arXiv:2607.10162 v2 ; lus : résumé, introduction), Jeong 2025
(arXiv:2511.10045 v6, AAAI-26 ; lus : résumé, introduction).

Hors tableau, lues en résumé seulement (« [résumé moteur] ») : Senna 2024
(DOI:10.1016/j.isci.2024.109167), Senna 2026 (DOI:10.53841/bpscog.2026.1.11.32),
Fryer 2014 (DOI:10.1016/j.cognition.2014.03.015), Hamilton-Fletcher 2018
(DOI:10.1016/j.cognition.2018.02.023), Bottini 2019
(DOI:10.1016/j.cognition.2019.01.006), Vogelsang 2025 (DOI:10.1111/desc.70099),
Achille 2019 (arXiv:1711.08856), Kleinman 2023 (arXiv:2308.12221), Zhang 2023
(arXiv:2312.00935), Plantinga 2026 (arXiv:2608.25976), Röder 2020
(DOI:10.1016/j.neubiorev.2020.10.030), Jakowski 2026
(DOI:10.1117/12.3094028, SPIE, accès fermé, résumé via S2), Phipson & Smyth
(arXiv:1603.05766).

## 3. Déjà publié

Chaque ligne donne au moins deux sources de nature différente, sinon elle porte
« à recouper ».

1. **Les réseaux profonds ont des périodes critiques d'intégration
   multisensorielle.** Article arXiv:2210.04643 (lu en entier), sa version
   publiée CVPR 2023 (DOI:10.1109/CVPR52729.2023.02327, même texte) ; dépôt de
   code officiel `mjkleinman/CriticalPeriodMultiView` (README : « Official
   code », Apache-2.0 ; contient les expériences CIFAR-10 / ResNet, ses journaux
   et ses scripts de tracé). Tient par la règle (article + code), **mais la
   réserve se renforce au tour 2** : le tour 1 citait arXiv:2506.15954 (Fukase
   2025) comme article indépendant ; le PDF ouvert montre que c'est une méthode
   pour repérer les périodes critiques de régularisation et d'élagage (« green
   AI »), qui cite Kleinman sans mener d'expérience multimodale. Il ne tient
   donc pas lieu de réplication de l'intégration multisensorielle. Les
   arXiv:1711.08856 et arXiv:2308.12221 viennent du même groupe. Aucune
   réplication indépendante du résultat multisensoriel n'a été trouvée sur
   6 canaux (journal, tour 2).
2. **Un contraste « déficit précoce contre déficit tardif » existe déjà dans un
   réseau artificiel.** Deux énoncés à séparer.
   2a. *Réseau à une seule modalité ou sans lésion de modalité* : l'effet
   précoce/tardif est publié par le groupe d'Achille (arXiv:1711.08856,
   [résumé moteur]) et retrouvé hors de ce groupe par Jakowski et
   Santos-Villalobos 2026 (DOI:10.1117/12.3094028, déficits sémantiques sur
   CIFAR-10, [résumé moteur], accès fermé) et par un dépôt individuel non
   évalué, `haeliotang/critical-period-lm` (README lu : essai auto-déclaré
   préenregistré, non évalué, sur le pré-entraînement de modèles de langue ; le contraste précoce/tardif y
   est mesuré avec le verdict `REVERSE_ONSET_EFFECT`, c'est-à-dire un dommage
   précoce initialement plus grand mais qui rétrécit plus vite que le tardif).
   Deux natures (article, dépôt) hors groupe : tient pour 2a, avec la réserve
   que ni l'un ni l'autre n'est multisensoriel.
   2b. *Réseau multisensoriel, déficit précoce contre tardif* : article
   arXiv:2210.04643, Fig. 8. **À recouper** : le dépôt de Kleinman lu au
   tour 2 (commit 3e86c6c, 2023-12-17) ne contient ni le Multi-View
   Transformer, ni les données vidéo, ni l'objectif de masquage (recherche de
   « video », « mask », « reconstruct », « transformer » dans les fichiers
   `.py` du dépôt, hors `deprecated/` : aucune expérience correspondante ;
   `main.py` n'accepte que les architectures `sresnet`, `sresnetmulti` et
   `nlayer`, définies dans `models.py` comme des ResNet et un discriminateur
   convolutif). La Fig. 8 repose donc sur le seul article. Fait :
   `docs/paper/main.tex:1100-1103` écrit « We are not aware of an
   artificial-network model that directly contrasts congenital and late sensory
   loss ». Inférence : cette phrase est fragilisée par la Fig. 8 de Kleinman,
   même si le déficit y est symétrique (décorrélation de deux images
   consécutives) et non une lésion d'une seule modalité.
3. **L'accord inter-culturel bouba/kiki est d'environ 72 %, pas ≥ 95 %.**
   Article DOI:10.1098/rstb.2020.0390 (lu en entier) ; données OSF w7crs
   (dossiers data, scripts, audio vérifiés, non réanalysés). Origine du 95 %
   retrouvée au tour 2 : Verhoef et al. 2024 (arXiv:2407.17974, PDF, page 1)
   écrivent que « 95 % » des humains nomment la figure anguleuse Kiki en citant
   Ramachandran et Hubbard 2001 (non lu) ; le chiffre vient donc d'un
   échantillon d'origine, pas de la comparaison de 25 langues. Fait :
   `docs/paper/main.tex:118` et `:1110-1111` annoncent « ≥ 95 % cross-cultural
   agreement ». Inférence : ce chiffre doit être sourcé (Ramachandran et
   Hubbard 2001, à lire) ou corrigé en 72 % [IC 56-82].
4. **Les modèles vision-langage montrent un effet bouba/kiki en zéro coup :
   contesté.** Pour : arXiv:2310.16781 (Alper, lu en entier, NeurIPS 2023) ;
   code `TAU-VAILab/kiki-bouba`. Contre ou nuancé : arXiv:2507.10013
   (Kouwenhoven 2025, NeurIPS 2025, lu §1-6, voir §2) ne trouve aucun effet
   cohérent sur CLIP ResNet et ViT ; arXiv:2407.17974 (Verhoef 2024, passages)
   « ne trouve pas de preuve concluante » sur quatre VLM ; Iida et Funakura
   2024 (VLM japonais, sans effet robuste ; cité par Kouwenhoven, non lu).
   Natures différentes (articles évalués par les pairs, sources et protocoles
   distincts) : la ligne tient dans sa forme « contesté », pas dans sa forme
   « montrent un effet ». Les auteurs de Kouwenhoven soutiennent que la méthode
   d'embeddings d'Alper peut mesurer des régularités propres à une langue plutôt
   qu'une association sensorielle ; c'est leur lecture, pas un fait établi.
5. **Une association son-forme mesurée sur modèles audio : résultats
   hétérogènes.** arXiv:2409.12306 (Tseng 2024, aussi publié à SLT 2024,
   DOI:10.1109/SLT61566.2024.10832254 ; tableau 2 lu) : séparation au-dessus du
   hasard pour les modèles entraînés sur parole (SpeechCLIP AUC phonétique
   0,87, FaST-VGS 0,84), sous le hasard pour AV-HuBERT (0,42) et CAV-MAE
   (0,31). arXiv:2607.10162 (Tsai 2026, résumé et introduction) : les modèles
   de parole alignent mal leurs jugements sur les humains et les modèles à poids
   ouverts ne relient pas de façon fiable un son entendu à sa forme. arXiv:2511.10045
   (Jeong, AAAI-26, résumé et introduction) : des intuitions phonétiques de
   MLLM alignées sur la linguistique, en texte, en alphabet phonétique (IPA) et en audio ; code
   `jjhsnail0822/sound-symbolism`. Jeux HF `Rapidata/kiki-bouba-audio-20k`
   (sans licence affichée), `dlion168/bouba-kiki-lalm` (cc-by-4.0) et, nouveau,
   `Rapidata/psychology-association-kiki-bouba-etc` (réponses humaines par
   paire d'images, sans licence affichée). Réserve : Kouwenhoven note que
   Tseng utilise la méthode d'embeddings d'Alper. Trois travaux de natures et
   d'équipes différentes : la ligne tient sous la forme « hétérogène » ; lecture
   partielle des trois (voir §2).
6. **Chez l'humain, la cécité tardive laisse une plasticité réelle mais non
   spécifique.** Article DOI:10.1093/brain/awt176 (lu en entier, IRMf + DCM) ;
   article DOI:10.1177/0956797619866625 (comportemental : LB intacts en SSA-H,
   lu en entier). Natures différentes (neuro-imagerie et comportement), même
   réseau de co-auteurs élargi (Röder/Collignon) : indépendance partielle.
7. **Une période sensible pour l'association son-forme chez les enfants nés
   aveugles puis opérés.** Contesté par les auteurs eux-mêmes : Sourav 2019
   conclut à une période sensible prolongée, Piller 2023 écrit avoir trouvé des
   « résultats opposés » (texte intégral PMC10349879, discussion). Deux tiers
   indépendants des deux groupes (Spence et Di Stefano 2024, note 7 ;
   Vreugdenhil 2025, introduction et discussion) lisent au contraire les deux
   études comme convergentes : l'effet bouba/kiki est plus faible, ou absent
   au début, chez les personnes aveugles ou opérées de cataracte, ce qui
   suggère un rôle de l'expérience visuelle. Les sources de nature différente
   (deux études primaires, deux revues ou discussions tierces) tiennent la
   phrase « la littérature est lue de deux façons » ; elles ne tranchent pas
   la période sensible. Voir l'arbitrage 3.1. Pas de consensus publié.

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
Tour 2, pour le contradicteur : **aucune source publiée lue ne rapporte ce
comptage.** Sourav 2019 (texte intégral PMC6787766, recherche par mots-clés : délai,
elapsed, since surgery, after surgery) ne donne ni délai entre opération et test, ni part des CC
testés au-delà de 6 mois. Piller 2023 note, dans sa discussion, que « tous les
individus testés dans 14 » ont été opérés plus de six mois avant le test, mais
la référence 14 est Pant et al. 2021 (illusion poids-taille, DOI:10.1038/s41598-021-86227-w),
pas Sourav 2019 (référence 33 de Piller) : ne pas confondre. Spence 2024 et
Vreugdenhil 2025 n'en disent rien. Le comptage de 29 CC sur 30 reste donc une
observation unique du tour 1, sans second témoin. Trois enregistrements OSF
figent les fichiers de Sourav depuis le 2019-08-16 (projet DOI:10.17605/OSF.IO/SQ2PY,
données DOI:10.17605/OSF.IO/VNZSU, matériel DOI:10.17605/OSF.IO/4W8PD) ; leur
résumé précise : « The project was not preregistered ». C'est un second témoin
de la phrase « étude non préenregistrée » (le texte de l'article est le
premier) ; il n'est pas vérifié que le fichier gelé `vnzsu` est identique à
celui compté au tour 1 (composant `79gn4/evpc2`).

**Voie 2 — tiers et revues.** Spence et Di Stefano 2024
(DOI:10.33735/phimisci.2024.11337), revue de la question de Molyneux, ouverte
au tour 2 (PDF, 21 pages ; lus : résumé, note 7, références). La note 7 écrit
que l'effet bouba/kiki « semble ne pas se produire » chez les aveugles
congénitaux (Sourav 2019), ni immédiatement chez les patients opérés de
cataracte, mais apparaître progressivement après la restitution de la vue
(Piller 2023) : une lecture convergente, sans mention d'un désaccord.
Vreugdenhil 2025 (DOI:10.1016/j.cortex.2025.08.003 ; lus : introduction,
discussion) cite Sourav 2019 et Piller 2023 avec Fryer 2014 et
Hamilton-Fletcher 2018 comme montrant des correspondances plus faibles chez
les personnes aveugles ou malvoyantes, d'où l'idée qu'une expérience visuelle
exacte et continue serait nécessaire : même lecture convergente. Les deux
tiers n'appartiennent à aucun des deux groupes (adresses du PDF : Oxford et
ISTC-CNR pour Spence et Di Stefano ; dépôts UvA et Radboud pour Vreugdenhil),
mais aucun n'a refait l'analyse. Fait : leur lecture ne signale
pas la contradiction que Piller signale lui-même. Inférence : le résumé
« Sourav oui, Piller non » du tour 1 compare des conclusions d'auteurs ; les
tiers comparent des directions d'effet. Senna 2024 et Senna 2026 viennent du
groupe de Piller : non indépendants. La littérature sur les aveugles
congénitaux va plutôt dans le sens d'un rôle de l'expérience visuelle (Fryer
2014, Hamilton-Fletcher 2018), avec un résultat contraire (Bottini 2019) ; ces
trois-là restent [résumé moteur].

Conclusion de l'arbitrage, à ce tour : la question « période sensible ou pas »
reste ouverte côté humain. Inférence : bouba_sens ne peut s'appuyer sur aucune
des deux études comme vérité de terrain ; il peut citer la divergence des
auteurs et la convergence de lecture des tiers. **À recouper** : (i) le
comptage de la voie 1 (contradicteur) ; (ii) les causes de l'écart Sourav /
Piller (âge au test, stimuli, population, essai unique) restent des
hypothèses non testées par une source.

## 4. Reste ouvert

1. **Lésion asymétrique d'une seule modalité, congénitale contre tardive, sur
   capteurs réels, avec lecture croisée de type Me6/B-3.** Preuve de l'absence
   à ce tour : arXiv `abs:"critical period" AND multimodal AND deprivation`
   = 0 résultat ; `abs:"critical learning period"` = 10 résultats, aucun avec
   lésion d'une seule modalité ; les 18 œuvres qui citent arXiv:2210.04643
   (S2) ne la font pas, d'après les titres et résumés ; Kleinman laisse
   l'asymétrie à des travaux futurs. **Tour 2, six canaux interrogés** (requêtes
   dans `veille-journal.md`) : OpenReview (37 requêtes sur le moteur de
   recherche, dont les groupes ICLR 2027, ICLR 2026, NeurIPS 2025 et ICLR 2025) :
   aucune soumission ne combine lésion d'une seule modalité et timing
   congénital/tardif ; les plus proches (ICLR 2027, titres seulement :
   « Understanding Missing-Modality Robustness through Target-Relevant
   Redundancy », « What Fusion Accuracy Hides About Missing-Modality
   Robustness ») traitent de modalités manquantes au test, pas d'un calendrier
   développemental. bioRxiv : balayage de l'API `details` de la catégorie
   neuroscience du 2024-01-01 au 2026-11-01 (231 DOI repérés par mots-clés, 8
   lus par le titre) et Europe PMC (19 préprints) : aucun modèle de réseau,
   aucun test bouba/kiki sur personnes aveugles ou opérées de cataracte.
   Thèses : HAL (9 requêtes) et theses.fr (9 requêtes) : une thèse de
   symbolisme sonore (De Carolis 2019), aucune sur le sujet. Code et données :
   GitHub (11 requêtes), Hugging Face, Zenodo, OSF : aucun dépôt de lésion
   asymétrique multimodale ; les plus proches sont des essais à une modalité ou
   sur modèles de langue (§3, ligne 2a). Fait : cinq natures de sources (archive
   de prépublications, graphe de citations, soumissions de conférences, thèses,
   dépôts de code) concordent sur l'absence. Réserve : la recherche plein texte
   d'OpenReview n'est pas déterministe (deux passes, 945 puis 1 118
   identifiants), les PDF et `notes?id=` d'OpenReview répondent 403 (une
   soumission ICLR 2027 non ouverte : « non vérifié », sa version arXiv
   arXiv:2609.36475 est déjà au tour 1), et un mot-clé manquant peut cacher un
   travail. Tient pour la couverture de ce tour, sous ces réserves.
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

État à la fin du tour 2. Cette note constate ; elle ne déclare pas le verrou
passé. Seul le PI le déclare, après le contradicteur.

| test | état à ce tour | ce qui manque |
|---|---|---|
| Saturation | Mesurée, non tranchée. Tour 2 : snowball de profondeur 1 sur 5 nouvelles graines (`snowball-tour2.tsv`, annoté dans `snowball-tour2.annot.tsv`) : 293 identifiants uniques, dont 240 absents du tour 1 (53 déjà dans le tour 1, dont un doublon S2 d'Achille 2018) (`snowball-tour1.tsv`). Titres lus : 7 nouvelles entrées directement pertinentes (touchent une phrase de la note : 2,9 % des 240), 18 de contexte (7,5 %), 215 hors sujet. Critère « < 5 % de nouveaux pertinents » : atteint si « pertinent » veut dire « touche une phrase » (7/240), non atteint si on compte aussi le contexte (25/240 = 10,4 %). La définition de « pertinent » n'est pas préenregistrée : le contradicteur arbitre. Le script plafonne la profondeur à 1 : la profondeur 2 depuis Kleinman n'a pas été faite. | Décision sur la définition de « pertinent » ; profondeur 2 depuis Kleinman 2022 et Sourav 2019 ; lecture des 7 entrées directes (Jakowski 2026, Iida 2024, Cuskley 2018, « Magnitude symbolism without sound », Huber 2015, McKyton 2015, Deroy 2013), toutes [résumé moteur] ou non ouvertes. |
| Recoupement | Partiel, en progrès. Marques « à recouper » : 4 au tour 1 (ligne 2, ligne 5, arbitrage 3.1, preuve de l'absence §4.1) → 3 à ce tour : (a) ligne 2b, Fig. 8 seulement dans l'article, le code public ne la reproduit pas ; (b) §3.1 voie 1, comptage OSF à refaire par le contradicteur, sans second témoin publié ; (c) §3.1, causes de l'écart Sourav/Piller, hypothèses non testées. Levées : ligne 2a (deux natures hors groupe), ligne 5 (devenue « hétérogène »), §4.1 (cinq natures de sources). Réserves qui ne sont pas des marques : la ligne 1 n'a plus de réplication indépendante (Fukase ne l'est pas) ; plusieurs lectures sont partielles (§2). | (a) code ou données de la Fig. 8 de Kleinman, ou un deuxième article multisensoriel ; (b) comptage par le contradicteur, données non randomisées de Sourav et Piller ; (c) une source qui teste les causes ; lecture complète de Tseng, Tsai, Jeong, Verhoef, Spence, Vreugdenhil ; Ramachandran et Hubbard 2001 pour le « 95 % ». |
| Couverture (6 canaux) | 6/6 interrogés (arXiv, OpenReview, bioRxiv, Crossref-journaux, thèses-HAL, code-données), requêtes consignées dans `veille-journal.md`. Limites : OpenReview non déterministe et PDF en 403 ; bioRxiv balayé sur la catégorie neuroscience seulement (le balayage toutes catégories, abandonné car trop lent, n'a couvert que 6 à 9 semaines sur environ 19 dans chacune de ses huit tranches) ; OpenAlex n'a pas été repris ; l'accès aux revues fermées (Brain, Psych. Sci. via PMC, SPIE) reste partiel. | Balayage bioRxiv des autres catégories (animal behavior and cognition, zoology) ; OpenAlex ; décision du contradicteur sur ce qu'il faut pour « interrogé ». |

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
| 2026-11-01 (tour 1) | DOI:10.33735/phimisci.2024.11337 | résumé seulement (anti-robot) ; rouvert au tour 2, voir ci-dessous |
| 2026-11-01 | Dépôt : amendement v0.7, ADR-0020, 0022, 0024, 0027, brouillon #66, `run_grid.sh`, `lesion.py`, `cli.py`, `loop.py` | lus en entier ou aux lignes citées |

Tour 2 (2026-11-01, soirée) :

| date | source | pages |
|---|---|---|
| 2026-11-01 | arXiv:2507.10013 v2 (NeurIPS 2025), PDF arXiv | corps du texte p. 1-10 ; annexes non lues |
| 2026-11-01 | arXiv:2409.12306 v3 (SLT 2024), PDF arXiv | résumé, §5.1-5.2 et tableau 2, début §6 |
| 2026-11-01 | arXiv:2607.10162 v2, PDF arXiv | résumé, introduction |
| 2026-11-01 | arXiv:2407.17974 v1, PDF arXiv | résumé, introduction |
| 2026-11-01 | arXiv:2511.10045 v6 (AAAI-26), PDF arXiv | résumé, introduction |
| 2026-11-01 | arXiv:2506.15954, PDF arXiv | résumé, introduction, recherche de mots-clés |
| 2026-11-01 | DOI:10.33735/phimisci.2024.11337, PDF éditeur (21 p.) | résumé, note 7, références ; recherche de mots-clés |
| 2026-11-01 | DOI:10.1016/j.cortex.2025.08.003, PDF déposé UvA (10 p.) | introduction, discussion, références ; recherche de mots-clés |
| 2026-11-01 | DOI:10.1177/0956797619866625 (PMC6787766), DOI:10.1038/s41598-023-38486-y (PMC10349879) | relus pour une recherche de mots-clés (délai opération-test, note 14 de Piller) |
| 2026-11-01 | `mjkleinman/CriticalPeriodMultiView`, commit 3e86c6c | README, `run_experiment.py`, `models.py`, `main.py` (options `--arch`), recherche de mots-clés dans les `.py` ; arXiv:2210.04643 §4.3-4.4 et légende de la Fig. 8 relus |
| 2026-11-01 | OSF registrations sq2py, vnzsu, 4w8pd (API OSF) | métadonnées et résumé de registre |
| 2026-11-01 | OpenReview (API de recherche) : fiches 2FaHRono5T, gkcU26BOml, vmG3EVg8Q5, gNrerlEU5F, nMsMyhFkDV, zUtl4kJa0C, qDFpNXnuYK | titre, résumé, lieu ; PDF et `notes?id=` en 403 |
| 2026-11-01 | `haeliotang/critical-period-lm`, `linusjoergensen/platonic-bouba-kiki-public` | début du README seulement (en-tête et premier tableau de résultats pour le premier) ; `jjhsnail0822/sound-symbolism` : existence seulement (lien du PDF de Jeong, liste GitHub), README non lu |

Non lus en détail à ce tour : ADR-0019 (deux fichiers), 0021, 0023, 0026.
