import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt

training_set = pd.read_csv("training_set.csv")

X_train = training_set.drop(columns=["Authenticated"])
y_train = training_set["Authenticated"]
reducer = umap.UMAP(random_state=42)

reduced = reducer.fit_transform(X_train)

plt.scatter(reduced[:,0],reduced[:,1], c = y_train)
plt.show()