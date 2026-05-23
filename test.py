import pandas as pd
train = pd.read_csv("data/train.csv")
print("Valores únicos en rating:", sorted(train['label'].unique()))
print("¿Hay nulos?", train['label'].isnull().sum())
print("Valor mínimo:", train['label'].min())