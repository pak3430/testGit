from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 연습용 데이터셋 로드
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    random_state=42)

# Estimator 인스턴스와 하이퍼 파라미터 설정
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

for pY, pT in zip(y_pred,y_test):
    print("[",pY,pT,"]")