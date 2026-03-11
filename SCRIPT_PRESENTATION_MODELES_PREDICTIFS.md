# Script de Présentation — Modèles Prédictifs
## HR Analytics – Prédiction de l'Attrition

---

### 1. Intro — Méthodologie
*(Ouvrir la page "Modèles Prédictifs")*

"Alors maintenant on passe à la partie prédiction. L'idée c'est simple : est-ce qu'on peut deviner à l'avance quel employé risque de partir ?

Pour ça on a testé trois modèles de Machine Learning :
- **Random Forest** — plein d'arbres de décision qui votent ensemble
- **Gradient Boosting** — des arbres qui se corrigent les uns après les autres
- **Régression Logistique** — un modèle plus simple, qui sert de référence

On a séparé les données en 80% pour entraîner et 20% pour tester. Et comme on a très peu de départs (16%), on a utilisé SMOTE pour rééquilibrer les classes."

---

### 2. Les KPIs
*(Montrer les 4 métriques en haut)*

"Ici on voit les résultats du meilleur modèle.

Juste pour expliquer rapidement les métriques :
- **Accuracy** c'est le taux de bonnes réponses global
- **Precision** c'est : quand le modèle dit 'va partir', est-ce qu'il a raison ?
- **Recall** c'est : parmi ceux qui sont vraiment partis, combien on en a détecté ? C'est la plus importante pour nous
- **F1-Score** c'est l'équilibre entre les deux"

---

### 3. Comparaison des modèles
*(Montrer barres + radar)*

"Le graphe à barres compare les trois modèles côte à côte. Le radar à droite donne une vue globale — plus la forme est grande, mieux c'est.

On voit que les trois sont proches, avec un AUC autour de 0.80 — c'est un bon score."

---

### 4. Validation croisée
*(Montrer le boxplot)*

"On se pose la question : est-ce que ces résultats sont fiables ou c'est juste de la chance ?

La validation croisée répond à ça — on teste le modèle 5 fois sur des parties différentes des données. Si les scores sont stables, c'est que le modèle a vraiment appris quelque chose. Et c'est le cas ici."

---

### 5. Importance des variables
*(Montrer les barres horizontales RF + GB)*

"C'est la partie la plus intéressante pour les RH.

Les deux modèles sont d'accord sur les facteurs principaux :
1. **OverTime** — le facteur n°1, les heures sup multiplient le risque par 3
2. **MonthlyIncome** — les bas salaires poussent les gens à partir
3. **Age / Ancienneté** — les jeunes nouvelles recrues sont les plus volatils
4. **StockOptionLevel** — pas de stock options = pas de raison de rester longtemps
5. **Satisfaction** — une baisse de satisfaction, c'est un signal d'alerte"

---

### 6. Courbes ROC
*(Montrer les courbes)*

"La courbe ROC mesure si le modèle fait mieux que le hasard. La diagonale grise c'est le hasard. Nos modèles sont bien au-dessus avec un AUC de 0.80 — donc dans 80% des cas, le modèle distingue correctement un futur départ d'un maintien."

---

### 7. Matrices de confusion
*(Montrer les 3 matrices)*

"La matrice de confusion montre le détail : combien de bonnes et de mauvaises prédictions.

Ce qui nous inquiète le plus c'est les faux négatifs en bas à gauche — c'est un départ qu'on n'a pas vu venir. Un faux positif c'est juste un entretien en plus, c'est pas grave. Mais rater un vrai départ, ça coûte cher."

---

### 8. Distribution des scores
*(Montrer l'histogramme)*

"Ce graphe montre comment le modèle note chaque employé. En vert ceux qui sont restés, en rouge ceux qui sont partis. On voit que le modèle arrive bien à les séparer — les verts sont à gauche (faible risque) et les rouges à droite (haut risque)."

---

### 9. Conclusion

"Pour résumer : le modèle marche bien, AUC de 0.80.

Les recommandations concrètes pour les RH :
1. **Limiter les heures sup** — c'est le levier n°1
2. **Revoir les salaires** des niveaux bas
3. **Accompagner les jeunes recrues** avec du mentorat
4. **Proposer des stock-options** même au niveau 1
5. **Surveiller la satisfaction** avec des enquêtes régulières

Et on garde en tête que le modèle est un outil d'aide, pas un système automatique — c'est toujours un humain qui décide de l'action."

---

### Transitions

- **Avant** : "Maintenant qu'on a exploré les données, est-ce qu'on peut prédire l'attrition ?"
- **Après** : "Avec ces modèles, on peut maintenant faire des prédictions par employé — c'est la section suivante."

---

### Questions possibles

**Pourquoi pas du deep learning ?**
"Avec 1 470 lignes, le deep learning risque le surapprentissage. Les modèles classiques marchent mieux ici et sont plus interprétables."

**C'est fiable avec si peu de départs ?**
"Oui, la validation croisée le confirme. Et on a utilisé SMOTE pour compenser le déséquilibre."

**Pourquoi OverTime est si important ?**
"30% des employés en heures sup partent, contre 10% pour les autres — c'est un risque 3 fois plus élevé."
