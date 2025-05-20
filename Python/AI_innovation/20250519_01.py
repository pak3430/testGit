import seaborn as sns
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt

# 데이터 로드
# data = sns.load_dataset("titanic")

# # jointplot
# sns.jointplot(data=data, x="sepal_length", y="sepal_width", kind="kde")

# plt.suptitle("JointPlot of Sepal Width vs Length", y=1.02)
# plt.show()

# sns.pairplot(data, hue="species",markers=["o","s","D"])
# plt.title("Iris Pair Plot, Hue로 꽃의 종을 시각화")
# plt.show()


titanic = sns.load_dataset("titanic")

print(titanic.head())
titanic.age = titanic.age.dropna()
temp = titanic.age.copy()
for a, b in temp, titanic.age:
    print(a)
    a= int(b)

titanic_size = titanic.pivot_table(index="survived",
                                   columns="age",
                                   aggfunc="size"
                                   )
sns.heatmap(titanic_size,
            cmap=sns.light_palette("red",as_cmap=True),
            annot=True,
            fmt="d"
            )
plt.title("titanic")
plt.show()