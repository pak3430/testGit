import seaborn as sns
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")

print(titanic.head())
titanic.age = titanic.age.dropna()  # Remove NaN values from 'age'
# The following loop has issues, I'll provide a corrected version if needed.
# temp = titanic.age.copy()
# for a, b in zip(temp, titanic.age):  # Use zip to iterate through both Series
#     print(a)
#     a = int(b)  # This assignment doesn't modify the original Series
# Instead of the loop, no need to convert age to int, pivot_table will handle it

titanic_size = titanic.pivot_table(index="survived",
                                    columns=pd.cut(titanic['age'], bins=10), # Group age into bins
                                    aggfunc="size"
                                    )
import pandas as pd
sns.heatmap(titanic_size,
            cmap=sns.light_palette("red", as_cmap=True),
            annot=True,
            fmt="d"
            )
plt.title("Titanic Survival by Age")
plt.xlabel("Age Group")
plt.ylabel("Survived")
plt.show()