"""
Scikit-learnを用いたIrisデータセットの読み込みと決定木（DecisionTreeClassifier）モデルの作成例
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report


def main():
    # 1. Irisデータセットの読み込み
    iris = load_iris()
    X, y = iris.data, iris.target

    print("特徴量名:", iris.feature_names)
    print("クラス分類ラベル:", iris.target_names)
    print(f"データ数: {X.shape[0]}, 特徴量数: {X.shape[1]}\n")

    # 2. 訓練データとテストデータに分割 (例: 80% 訓練, 20% テスト)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 決定木分類器（DecisionTreeClassifier）のインスタンス化
    # 過学習を防ぎたい場合は max_depth などのハイパーパラメータを指定可能
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)

    # 4. モデルの学習
    clf.fit(X_train, y_train)

    # 5. テストデータを用いた予測
    y_pred = clf.predict(X_test)

    # 6. モデルの評価
    acc = accuracy_score(y_test, y_pred)
    print(f"テストデータの正解率 (Accuracy): {acc:.4f}\n")

    print("分類レポート:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 7. 学習した決定木のルール（テキスト形式）を出力
    tree_rules = export_text(clf, feature_names=list(iris.feature_names))
    print("決定木の分岐ルール:")
    print(tree_rules)


if __name__ == "__main__":
    main()
