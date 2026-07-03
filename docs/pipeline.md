# Notes sur le pipeline

Ce document resume le role de chaque etape dans le pipeline du projet.

## 1. Reconstruction avec VGGT-Omega

VGGT-Omega est utilise pour remplacer l'etape de reconstruction en amont
originalement utilisee par GaussGym. A partir d'une video monoculaire, des
images sont extraites puis traitees par VGGT-Omega afin d'estimer :

- les intrinseques camera ;
- les extrinseques camera ;
- les cartes de profondeur ;
- les cartes de confiance de profondeur ;
- les cartes de points 3D denses.

Ces sorties sont ensuite utilisees dans deux branches :

- une branche geometrique pour NKSR ;
- une branche photorealiste pour GSplat.

## 2. Maillage physique avec NKSR

NKSR reconstruit un maillage triangulaire continu a partir d'un nuage de points
oriente. Le nuage de points doit contenir des points 3D et des normales fiables.
Dans le travail TER, les normales sont estimees depuis les points VGGT-Omega avec
une strategie de PCA locale, car cette methode etait plus robuste que les
normales calculees directement par produit vectoriel entre voisins dans la carte
de profondeur.

Le maillage obtenu est utilise comme geometrie de collision dans GaussGym.

## 3. Rendu photorealiste avec GSplat

GSplat reconstruit l'apparence visuelle de la scene sous la forme d'un ensemble
de gaussiennes 3D. Il utilise les images et les parametres camera estimes par
VGGT-Omega, convertis dans une structure compatible COLMAP.

GSplat est utilise uniquement pour le rendu RGB photorealiste. Il ne fournit pas
la geometrie de collision physique.

## 4. Integration dans GaussGym

GaussGym combine :

- le maillage NKSR pour la physique ;
- la scene GSplat pour les observations RGB.

Le principal probleme d'integration est l'alignement des coordonnees. GSplat
normalise la scene COLMAP pendant l'entrainement, donc le splat exporte est
exprime dans un repere normalise. Le maillage NKSR et les trajectoires camera
GaussGym doivent donc etre transformes vers ce meme repere normalise.

Le script `gaussgym/adapt_meshes_to_gsplat_normalization.py` effectue cette
conversion.
