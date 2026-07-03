# Pipeline Real-to-Sim : VGGT-Omega, NKSR, GSplat et GaussGym

Ce depot contient les scripts developpes pendant un projet TER sur la
reconstruction 3D photorealiste et physiquement coherente d'environnements a
partir de videos monoculaires.

L'objectif est de relier quatre outils de recherche existants dans un meme
pipeline Real-to-Sim :

- VGGT-Omega pour la reconstruction 3D a partir d'images extraites d'une video ;
- NKSR pour la reconstruction du maillage physique ;
- GSplat pour le rendu photorealiste par Gaussian Splatting ;
- GaussGym pour la simulation robotique avec Isaac Gym.

Ce depot ne redistribue pas ces projets externes, leurs modeles preentraines,
les gros jeux de donnees, les fichiers Gaussian Splat, les scenes reconstruites
ou les binaires Isaac Gym. Ils doivent etre installes separement depuis leurs
sources officielles.

Seuls les scripts d'adaptation developpes pour ce projet sont fournis ici.

## Contexte du projet

GaussGym est un framework Real-to-Sim modulaire qui separe la representation
utilisee pour la physique de celle utilisee pour le rendu visuel. Dans le
pipeline etudie ici, la geometrie physique est representee par un maillage
genere avec NKSR, tandis que les observations RGB photorealistes sont generees
avec GSplat.

Les exemples originaux de GaussGym reposent principalement sur des scenes deja
reconstruites, par exemple a partir de scans de smartphone ou de donnees
Polycam. Dans ce projet, cette etape de reconstruction en amont est remplacee
par VGGT-Omega afin de partir d'une simple video monoculaire.

Le pipeline complet est le suivant :

1. Extraire les images d'une video monoculaire.
2. Executer VGGT-Omega pour estimer les parametres camera, les cartes de
   profondeur, les cartes de confiance et les points 3D denses.
3. Convertir les sorties de VGGT-Omega vers une structure compatible COLMAP pour
   GSplat.
4. Preparer un nuage de points oriente pour NKSR.
5. Reconstruire un maillage physique avec NKSR.
6. Entrainer ou exporter une scene GSplat photorealiste.
7. Aligner le maillage NKSR et les trajectoires camera GaussGym avec le repere
   normalise de GSplat.
8. Charger la scene adaptee dans GaussGym.

## Dependances externes

Installez les projets principaux depuis leurs depots officiels ou leurs
instructions d'installation officielles :

- VGGT-Omega
- NKSR
- GSplat
- GaussGym
- NVIDIA Isaac Gym
- COLMAP ou une structure de donnees compatible COLMAP

Les scripts de ce depot supposent que ces outils sont deja installes et
disponibles dans vos environnements Python.

## Structure du depot

```text
vggt_omega/
  extract_omega_from_video_gsplat.py

nksr/
  run_nksr_from_vggt_omega.py

gaussgym/
  adapt_meshes_to_gsplat_normalization.py

docs/
  pipeline.md
```

## Scripts

### 1. Preparation des donnees VGGT-Omega pour GSplat

Fichier :

```text
vggt_omega/extract_omega_from_video_gsplat.py
```

Ce script execute VGGT-Omega sur des images extraites d'une video et prepare les
sorties pour les etapes suivantes du pipeline.

Il sert a :

- charger un checkpoint VGGT-Omega ;
- extraire les images d'une video monoculaire ;
- estimer les intrinseques et extrinseques camera ;
- predire les cartes de profondeur et les cartes de confiance ;
- reconstruire des points 3D denses a partir de la profondeur ;
- verifier la convention des poses camera ;
- filtrer les points peu fiables ;
- preparer les donnees utilisables pour l'entrainement GSplat et le
  pretraitement NKSR.

Dans le rapport TER, cette partie correspond a l'adaptation des sorties de
VGGT-Omega pour la branche geometrique et la branche photorealiste du pipeline.

### 2. Reconstruction du maillage avec NKSR

Fichier :

```text
nksr/run_nksr_from_vggt_omega.py
```

Ce script reconstruit un maillage physique a partir du nuage de points et des
normales prepares depuis les sorties VGGT-Omega.

Il sert a :

- charger un ou plusieurs fichiers `.npz` contenant points, normales et
  couleurs ;
- filtrer les points et normales invalides ;
- sous-echantillonner optionnellement le nuage de points ;
- executer la reconstruction NKSR sur GPU ;
- extraire un maillage triangulaire ;
- transferer les couleurs du nuage de points source vers les sommets du
  maillage ;
- exporter le resultat sous le nom `nksr_mesh.ply`.

Ce maillage est destine a etre utilise comme geometrie de collision physique
dans GaussGym.

### 3. Alignement des coordonnees GaussGym / GSplat

Fichier :

```text
gaussgym/adapt_meshes_to_gsplat_normalization.py
```

C'est le script d'integration principal pour GaussGym.

Le probleme principal resolu par ce script est le decalage de repere entre :

- le maillage NKSR, initialement exprime dans le repere de reconstruction
  original ;
- les trajectoires camera GaussGym ;
- la scene GSplat exportee, exprimee dans le repere COLMAP normalise de GSplat.

La solution choisie consiste a ne pas modifier le Gaussian Splat exporte. A la
place, le script transforme les fichiers de scene GaussGym afin que les sommets
du maillage et les trajectoires camera soient exprimes dans le meme repere
normalise que GSplat.

Le script :

- lit la transformation de normalisation du parser GSplat/COLMAP ;
- transforme les sommets du maillage GaussGym ;
- transforme les positions camera (`cam_trans`) ;
- transforme les orientations camera (`cam_quat`) ;
- remet `offset` a zero ;
- remet `from_ig_rotation` a l'identite ;
- ecrit un fichier `splatfacto/dataparser_transforms.json` identite, car le
  splat est deja normalise.

Cette etape est necessaire avant de charger ensemble le maillage NKSR et le rendu
GSplat dans GaussGym.

## Exemple d'utilisation

Les chemins exacts dependent de votre installation locale. Les commandes
suivantes montrent l'ordre d'utilisation prevu.

### Executer VGGT-Omega et preparer les donnees GSplat/NKSR

Modifiez les chemins dans :

```text
vggt_omega/extract_omega_from_video_gsplat.py
```

Puis executez le script dans votre environnement VGGT-Omega :

```bash
python vggt_omega/extract_omega_from_video_gsplat.py
```

Le script produit les sorties VGGT-Omega : parametres camera, cartes de
profondeur, cartes de confiance, nuages de points et donnees converties pour les
etapes suivantes.

### Executer NKSR

Dans votre environnement NKSR :

```bash
python nksr/run_nksr_from_vggt_omega.py \
  --inputs /path/to/nksr_input_pca.npz \
  --output_dir /path/to/nksr_outputs \
  --max_points 300000 \
  --detail_level 1.0 \
  --mise_iter 1
```

Le maillage de sortie est sauvegarde sous le nom :

```text
nksr_mesh.ply
```

### Aligner la scene GaussGym avec GSplat

Dans l'environnement GaussGym :

```bash
python gaussgym/adapt_meshes_to_gsplat_normalization.py \
  --scene-dir /path/to/original_gaussgym_scene \
  --out-scene-dir /path/to/aligned_gaussgym_scene \
  --gsplat-root /path/to/gsplat \
  --data-dir /path/to/colmap_data \
  --overwrite
```

La scene de sortie peut ensuite etre chargee dans GaussGym comme scene locale.

Exemple :

```bash
gauss_play \
  --runner.load_run=<RUN_NAME> \
  --terrain.scenes.max_num_scenes=1 \
  --terrain.scenes.iphone_data.repo_id=local:/path/to/aligned_gaussgym_scene \
  --terrain.scenes.iphone_data.scene='' \
  --env.force_renderer=True
```

## Notes

Ce depot est une publication legere du code permettant de reproduire le travail
d'adaptation. Ce n'est pas une implementation autonome complete de VGGT-Omega,
NKSR, GSplat ou GaussGym.

Les gros fichiers comme les checkpoints, videos, maillages reconstruits,
fichiers PLY de Gaussian Splat et logs de simulation ne doivent pas etre ajoutes
a ce depot.
