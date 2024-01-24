模様のcube解決
可換の範囲を増やす
coset分解使える？
　効率のいい分解を見つけないとこれ以上は厳しそう
　2/2/2と3/3/3あたりで感覚をつかむ
問題分割
　ヒューリスティックにやる 
　subgroup chain
完　まだやってないglobe解決
いったん完　既存の解をshrinkさせる

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

3x3x3 中央揃えたら後は0と2だけでよい 探索空間減らせる

publicになっているもの
https://www.kaggle.com/code/yunsuxiaozi/summary-of-mainstream-algorithms-santa-2023汎用cubesolver wreath 5,7,12の最適解
wildcardのために早期修了できないかチェック 同じ状態を２回通っていないかチェック 適当に操作列を取って、より短い長さで置き換えられないかチェック

n 解の長さ
２状態間のshortest_pathを記録する
基本100

globeは以下の２ステップに分けてよいか？

1. 隣を同じ文字にする
2. 順番を適切に入れ替える

ラストの何手かは読めるので、最後の方だけ見て数手減らせるかも

globe考察
https://www.jaapsch.net/puzzles/master.htm

1. とりあえず１つずつ揃えれば右下は揃う
2. そのまま右上も揃う
3. 後どうするか

1. 同じものが隣になるようにする
2. 内部の入れ替え


https://www.universiteitleiden.nl/binaries/content/assets/science/mi/scripties/bachbrouwer.pdf

globe
1. 上下の組を作る 
   1. 反対側に置いてあるものを探す 
   2. 移動幅は1,3,5,...,n-1　これを偶数個組み合わせる 
   3. フリップが偶数でなければやり直す 
2. 上下のflipを直す 
   1. 反対側まで移動してフリップする 
3. 必要なら中央を解く　書き出し必要 
4. 左右の順番入れ替える　隣接入れ替えと二乗誤差で山登り法できる？
展開図で表示

globeのアルゴリズム実装する
wreathのアルゴリズム実装する
wreathのwildcard考慮
globeのwildcard考慮
cubeのwildcard考慮

wreath
　２行　中央　２行　で表示
　-l.r.l
　　右の輪（接点以外）と上の接点左と下の接点右上が反時計回りに１つずれる
　-l.-r.l
　　右の輪（接点以外）と下の接点右上と上の接点左が時計回りに１つずれる
　r.l.-r
　　左の輪（接点以外）と下の接点左上と上の接点右が反時計回りに１つずれる
　r.-l.-r
　　左の輪（接点以外）と上の接点右と下の接点左上が時計回りに１つずれる

　同じ文字を一定距離で配置して接点で並ぶようにする
　少しずつ入れ替えていく

cube
　3x3x3に帰着する過程をもう少し理解しておきたい（優先度低い）
　180度回転が２手になること、向きが決まっていることを考慮する
　展開図で表示

visualizer
　o 起動後に問題を指定して読み込み
　o 手を１つずつ入れる
　sugarは作っておく
1. cube　180度回転
2. cube　全体的に回転　右　左　上　下
3. o globe　行（上半分）のリストと列を指定して左右入れ替え
4. o globe　行（上半分）のリストを指定して対角線上の入れ替え
5. globe　行（下半分）を指定して対角線の取り込み
　o 元に戻す/やり直す コマンドを準備　u,r
　o 今までの手を表示するコマンドを準備　s
　o ドットで区切って入力できる
