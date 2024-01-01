小さい結果の最適解は大きい問題にも適用できる
部分的な解を組み合わせてなんとかならないか
既存解と等価で短いものを作れないか

既存解を可視化したい
pygame?

可換なものを調整
複数手順の同一視

幅優先探索による全列挙
wreath_6,7,12
cube_2/2/2
状態をグラフで定義して最短距離

Schreier–Sims algorithm

パズル状態を順列で定義し、 あり得る状態を集めた群を考える  
操作は群の１要素から１要素へのmappingとなる  
群関係のアルゴリズムを適用できる？

がんばってこの辺を読む?
https://github.com/dwalton76/rubiks-cube-NxNxN-solver/blob/master/rubikscubennnsolver/RubiksCubeNNNEven.py

3x3x3 中央揃えたら後は0と2だけでよい
　探索空間減らせる

publicになっているもの
https://www.kaggle.com/code/yunsuxiaozi/summary-of-mainstream-algorithms-santa-2023
　汎用cubesolver
　wreath 5,7,12の最適解
　wildcardのために早期修了できないかチェック
　同じ状態を２回通っていないかチェック
　適当に操作列を取って、より短い長さで置き換えられないかチェック

n 解の長さ
２状態間のshortest_pathを記録する
基本100

globeは以下の２ステップに分けてよいか？
1. 隣を同じ文字にする
2. 順番を適切に入れ替える

ラストの何手かは読めるので、最後の方だけ見て数手減らせるかも
