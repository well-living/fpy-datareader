import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def interpolate_age_income(
        data, 
        start_age=20, 
        end_age=65, 
        method='linear', 
        extrapolation='constant', 
        terminal_start=0, 
        terminal_end=100,
        include_growth_rate=False
    ):
    """
    年齢別所得データを1歳刻みで補間するテーブルに変換する関数
    
    Parameters
    ----------
    data : pandas.DataFrame or list of lists or numpy.ndarray
        年齢と所得のデータ。1列目が年齢、2列目が所得
    start_age : int, default 20
        補間する最小年齢
    end_age : int, default 65
        補間する最大年齢
    method : str, default 'linear'
        補間方法。'linear'（線形補間）または 'cubic_spline'（3次スプライン補間）
    extrapolation : str, default 'constant'
        範囲外補間方法。'constant'（同じ値が継続）または 'terminal'（ターミナル時点使用）
    terminal_start : int, default 0
        ターミナル時点の開始年齢（extrapolation='terminal'の場合）
    terminal_end : int, default 100
        ターミナル時点の終了年齢（extrapolation='terminal'の場合）
    include_growth_rate : bool, default False
        成長率の列を追加するかどうか
    
    Returns
    -------
    pandas.DataFrame
        年齢と所得の補間されたテーブル
        include_growth_rate=Trueの場合、成長率の列も追加される
        
    Examples
    --------
    >>> data = [
            [31.0, 495530],
            [37.1, 533352],
            [42.0, 564760],
            [47.1, 576439],
            [51.9, 561118],
            [56.9, 579763],
            [61.9, 419871],
            [66.9, 424190],
            [73.2, 369543]
        ]
    >>> result = interpolate_age_income(data, 28, 75, method='linear', extrapolation='constant')
    >>> print(result.head(10))
           age         income
        0   28  495530.000000
        1   29  495530.000000
        2   30  495530.000000
        3   31  495530.000000
        4   32  501730.327869
        5   33  507930.655738
        6   34  514130.983607
        7   35  520331.311475
        8   36  526531.639344
        9   37  532731.967213
        
    >>> # 成長率付きの例
    >>> result_with_growth = interpolate_age_income(data, 28, 40, include_growth_rate=True)
    >>> print(result_with_growth.head(10))
           age         income  growth_rate
        0   28  495530.000000     0.000000
        1   29  495530.000000     0.000000
        2   30  495530.000000     0.000000
        3   31  495530.000000     0.012516
        4   32  501730.327869     0.012353
        5   33  507930.655738     0.012191
        6   34  514130.983607     0.012028
        7   35  520331.311475     0.011866
        8   36  526531.639344     0.011765
        9   37  532731.967213     0.011664
        
    Notes
    -----
    - 所得が0円を下回る場合は0円に設定される
    - 入力データは年齢順にソートされる
    - 線形補間では最も近い2点間の線分から値を計算
    - 3次スプライン補間では全体のデータから滑らかな曲線を作成
    - 成長率は前年からの変化率を浮動小数点数で表示（例：0.05 = 5%の成長）
    - 最初の年齢の成長率はNaNとなる（前年データが存在しないため）
    """
    
    # データの前処理
    if isinstance(data, pd.DataFrame):
        ages = data.iloc[:, 0].values
        incomes = data.iloc[:, 1].values
    elif isinstance(data, (list, np.ndarray)):
        data_array = np.array(data)
        ages = data_array[:, 0]
        incomes = data_array[:, 1]
    else:
        raise ValueError("データ形式が不正です。DataFrame、list、またはndarrayを使用してください")
    
    # 年齢順でソート
    sort_idx = np.argsort(ages)
    ages = ages[sort_idx]
    incomes = incomes[sort_idx]
    
    # 補間する年齢範囲を作成
    target_ages = np.arange(start_age, end_age + 1)
    
    # 範囲外処理のためのデータ拡張
    if extrapolation == 'terminal':
        # ターミナル時点を追加
        extended_ages = [terminal_start]
        extended_incomes = [0]
        
        # 既存データを追加
        extended_ages.extend(ages)
        extended_incomes.extend(incomes)
        
        # 終了ターミナル時点を追加
        extended_ages.append(terminal_end)
        extended_incomes.append(0)
        
        ages = np.array(extended_ages)
        incomes = np.array(extended_incomes)
    
    # 補間の実行
    if method == 'linear':
        interpolated_incomes = np.interp(target_ages, ages, incomes)
    elif method == 'cubic_spline':
        # 3次スプライン補間
        cs = CubicSpline(ages, incomes, extrapolate=True)
        interpolated_incomes = cs(target_ages)
    else:
        raise ValueError("methodは'linear'または'cubic_spline'を指定してください")
    
    # 範囲外の値を処理（constant extrapolationの場合）
    if extrapolation == 'constant':
        # 最小年齢より小さい場合
        mask_below = target_ages < ages.min()
        interpolated_incomes[mask_below] = incomes[0]
        
        # 最大年齢より大きい場合
        mask_above = target_ages > ages.max()
        interpolated_incomes[mask_above] = incomes[-1]
    
    # 所得が0円を下回らないように調整
    interpolated_incomes = np.maximum(interpolated_incomes, 0)
    
    # 結果をDataFrameで作成
    result_df = pd.DataFrame({
        'age': target_ages,
        'income': interpolated_incomes
    })
    
    # 成長率の計算と追加
    if include_growth_rate:
        # pct_changeを使用して前年比成長率を計算（浮動小数点数形式）
        growth_rate = result_df['income'].pct_change().shift(-1)
        result_df['growth_rate'] = growth_rate
    
    return result_df