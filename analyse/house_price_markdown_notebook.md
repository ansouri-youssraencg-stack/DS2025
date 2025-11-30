# 📘 Projet Complet : Prédiction du Prix des Maisons (House Price Regression)

Ce notebook en **Markdown** présente un pipeline complet de Machine Learning : EDA, preprocessing, modèles, optimisation et visualisation.

---

# 1. 🎯 Objectif du Projet
Prédire le prix des maisons (`House_Price`) à partir de leurs caractéristiques :
- Surface habitable
- Taille du garage
- Superficie du terrain
- Matériaux
- Localisation, etc.

Dataset utilisé : **Home Value Insights (Kaggle)**.

---

# 2. 📥 Chargement des Données
```python
import kagglehub
path = kagglehub.dataset_download("prokshitha/home-value-insights")
df = pd.read_csv(f"{path}/house_price_regression_dataset.csv")
```
Aperçu :
```python
df.info()
df.describe(include='all').T
```

---

# 3. 🧹 Nettoyage & Préparation
### ✦ Vérification des valeurs manquantes
```python
df.isnull().sum()
```
### ✦ Définition de la cible et features
```python
target = "House_Price"
X = df.drop(columns=[target])
y = df[target]
```
### ✦ Détection des types de colonnes
```python
num_cols = X.select_dtypes(include=["int64", "float64"]).columns\ ncat_cols = X.select_dtypes(include=["object", "category"]).columns
```

---

# 4. 🔍 Analyse Exploratoire (EDA)
### 📌 Distribution de la variable cible
Histogramme + KDE.

### 📌 Corrélations
- `Square_Footage` et `Garage_Size` → variables les plus corrélées au prix.
- Matrice de corrélation heatmap.

---

# 5. 🛠️ Préprocessing
### ✦ Pipelines
- Imputation des valeurs manquantes
- Encodage des variables catégorielles (OneHot)
- Standardisation des variables numériques

```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])
```

---

# 6. 🤖 Modèles de Régression
Modèles testés :
- **Régression Linéaire**
- **Lasso Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Pipeline global :
```python
model = Pipeline([
    ("preprocess", preprocessor),
    ("reg", RandomForestRegressor())
])
```

---

# 7. 📊 Validation Croisée
```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
```
Évaluation : MSE, RMSE, MAE, R².

---

# 8. 🔧 Optimisation des Hyperparamètres
Exemple pour Random Forest :
```python
params = {
    "reg__n_estimators": [100, 300],
    "reg__max_depth": [None, 10, 20],
}

grid = GridSearchCV(model, params, cv=3, scoring="neg_mean_squared_error")
grid.fit(X, y)
```

---

# 9. ⭐ Comparaison des Modèles
Tableau comparatif :
- RMSE
- MAE
- R²

Analyse : Gradient Boosting & Random Forest offrent les meilleures performances.

---

# 10. 🔍 Importance des Variables
Pour Random Forest :
```python
model.named_steps['reg'].feature_importances_
```
Interprétation :
- `Square_Footage` (le plus important)
- `Garage_Size`
- `Lot_Size`

---

# 11. 📈 Régression Linéaire & Logistique (Feature la plus corrélée)
### ✦ Régression linéaire : droite prédite
### ✦ Régression logistique : sigmoïde + matrice de confusion + ROC / AUC

Objectif : comprendre l’impact de la feature la plus corrélée.

---

# 12. 💾 Sauvegarde du Modèle Final
```python
joblib.dump(grid.best_estimator_, "house_price_model.joblib")
```
Le modèle est maintenant prêt pour la prédiction !

---

# 13. 🚀 Prédictions sur de Nouvelles Données
```python
model = joblib.load("house_price_model.joblib")
model.predict(new_data)
```

---

# ✔️ Conclusion
Ce projet couvre tout le cycle ML :
- EDA
- Nettoyage et préparation
- Pipelines
- Modèles variés
- Hyperparam tuning
- Visualisation
- Export du modèle

Prêt pour un déploiement ou intégration API.

