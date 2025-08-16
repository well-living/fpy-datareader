import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import pandas as pd

def extract_numbers(text):
    """文字列から数値を抽出する関数"""
    numbers = []
    current_number = ""
    
    for char in text:
        if char.isdigit():
            current_number += char
        else:
            if current_number:
                numbers.append(int(current_number))
                current_number = ""
    
    # 最後の数値を追加
    if current_number:
        numbers.append(int(current_number))
    
    return numbers

def calculate_age_midpoint(age_class_str, method='discrete', default_class_width=5):
    """
    年齢階級の文字列から中央値を計算する関数
    
    Parameters:
    age_class_str (str): 年齢階級を表す文字列
    method (str): 計算方法
        - 'discrete': 離散的解釈 (30～34) → (30+34)/2 = 32.0
        - 'continuous': 連続的解釈 (30～34) → (30+35)/2 = 32.5
    default_class_width (int): 階級幅が明記されていない場合のデフォルト値
    
    Returns:
    float: 年齢階級の中央値
    """
    # 全角数字を半角に変換
    normalized_str = age_class_str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    numbers = extract_numbers(normalized_str)
    
    # 階級幅を抽出（5歳階級、10歳階級など）
    class_width = default_class_width  # デフォルト値を設定
    if '歳階級' in age_class_str:
        # 「○歳階級」のパターンから階級幅を抽出
        for i, num in enumerate(numbers):
            # 数値の後に「歳階級」があるかチェック
            if str(num) + '歳階級' in normalized_str:
                class_width = num
                break
    
    # 「30歳未満」、「29歳以下」、「～19歳」のパターン
    if ('歳未満' in age_class_str or '歳以下' in age_class_str or 
        (age_class_str.startswith('～') and '歳' in age_class_str)) and numbers:
        
        if '歳未満' in age_class_str:
            # 30歳未満の場合、実際の上限は29歳
            upper_age = numbers[0] - 1  # 30 → 29
        elif '歳以下' in age_class_str or age_class_str.startswith('～'):
            # 29歳以下や～19歳の場合、上限はその年齢
            upper_age = numbers[0]  # 29 → 29 or 19 → 19
        
        lower_age = upper_age - class_width + 1  # 29 - 5 + 1 = 25
        
        if method == 'discrete':
            return float((lower_age + upper_age) / 2)  # (25+29)/2 = 27
        else:  # continuous
            # 連続的解釈: lower_age から (upper_age + 1) まで
            return float((lower_age + (upper_age + 1)) / 2)  # (25+30)/2 = 27.5
    
    # 「85歳以上」、「70歳～」のパターン
    elif ('歳以上' in age_class_str or 
          (age_class_str.endswith('歳～') or age_class_str.endswith('～'))) and numbers:
        
        # 最初の数値が下限値
        lower_age = numbers[0]  # 85 or 70
        upper_age = lower_age + class_width - 1  # 85 + 5 - 1 = 89
        
        if method == 'discrete':
            return float((lower_age + upper_age) / 2)  # (85+89)/2 = 87
        else:  # continuous
            # 連続的解釈: lower_age から (lower_age + class_width) まで
            return float((lower_age + (lower_age + class_width)) / 2)  # (85+90)/2 = 87.5
    
    # 「30～34」のようなパターン
    elif '～' in age_class_str and len(numbers) >= 2:
        start_age = numbers[0]
        end_age = numbers[1]
        
        if method == 'discrete':
            # 離散的解釈: (30+34)/2 = 32.0
            return float((start_age + end_age) / 2)
        else:  # continuous
            # 連続的解釈: (30+35)/2 = 32.5
            return float((start_age + (end_age + 1)) / 2)
    
    # パターンにマッチしない場合はNoneを返す
    return None

def add_age_midpoint_column(df, age_column_name, income_column_name, method='discrete', default_class_width=5):
    """
    データフレームに年齢階級の中央値列を追加する関数
    
    Parameters:
    df (pd.DataFrame): 対象のデータフレーム
    age_column_name (str): 年齢階級が入っている列の名前
    income_column_name (str): 所得が入っている列の名前
    method (str): 計算方法
        - 'discrete': 離散的解釈 
        - 'continuous': 連続的解釈
    default_class_width (int): 階級幅が明記されていない場合のデフォルト値
    
    Returns:
    pd.DataFrame: age(float)とincomeの2列のデータフレーム
    """
    result_df = pd.DataFrame()
    result_df['age'] = df[age_column_name].apply(
        lambda x: calculate_age_midpoint(x, method, default_class_width)
    )
    result_df['income'] = df[income_column_name]
    return result_df



def interpolate_age_income(
        data, 
        start_age=20, 
        end_age=65, 
        method='linear', 
        extrapolation='constant', 
        terminal_start=0, 
        terminal_end=100,
        include_growth_rate=False,
        include_growth_ma=False,
        growth_ma_periods=3
    ):
    """
    年齢別所得データを1歳刻みで補間するテーブルに変換する関数
    
    Parameters
    ----------
    data : pandas.DataFrame or list of lists or numpy.ndarray
        年齢と所得のデータ。1列目が年齢、2列目が所得
    start_age : int, default 20
        補間する最小年齢（0以上）
    end_age : int, default 65
        補間する最大年齢（start_age以上）
    method : str, default 'linear'
        補間方法。'linear'（線形補間）または 'cubic_spline'（3次スプライン補間）
    extrapolation : str, default 'constant'
        範囲外補間方法。'constant'（同じ値が継続）または 'terminal'（ターミナル時点使用）
    terminal_start : int, default 0
        ターミナル時点の開始年齢（0以上）
    terminal_end : int, default 100
        ターミナル時点の終了年齢（terminal_start以上）
    include_growth_rate : bool, default False
        成長率の列を追加するかどうか
    include_growth_ma : bool, default False
        成長率の移動平均列を追加するかどうか（include_growth_rate=Trueの場合のみ有効）
    growth_ma_periods : int, default 3
        移動平均の期間数（1以上）
    
    Returns
    -------
    pandas.DataFrame
        年齢と所得の補間されたテーブル
        include_growth_rate=Trueの場合、成長率の列も追加される
        include_growth_ma=Trueの場合、成長率の移動平均列も追加される
        
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
        
    >>> # 成長率と移動平均付きの例
    >>> result_with_ma = interpolate_age_income(data, 28, 35, include_growth_rate=True, 
    ...                                        include_growth_ma=True, growth_ma_periods=3)
    >>> print(result_with_ma)
           age         income  growth_rate  growth_rate_ma
        0   28  495530.000000     0.000000        0.000000
        1   29  495530.000000     0.000000        0.000000
        2   30  495530.000000     0.000000        0.004172
        3   31  495530.000000     0.012516        0.008289
        4   32  501730.327869     0.012353        0.008353
        5   33  507930.655738     0.012191        0.012353
        6   34  514130.983607     0.012028        0.012191
        7   35  520331.311475     0.011866        0.012028
        
    Notes
    -----
    - 所得が0円を下回る場合は0円に設定される
    - 入力データは年齢順にソートされる
    - 線形補間では最も近い2点間の線分から値を計算
    - 3次スプライン補間では全体のデータから滑らかな曲線を作成
    - 成長率は前年からの変化率を浮動小数点数で表示（例：0.05 = 5%の成長）
    - 最初の年齢の成長率はNaNとなる（前年データが存在しないため）
    """
    
    # パラメータの制約チェック
    if start_age < 0:
        raise ValueError("start_ageは0以上である必要があります")
    if end_age < start_age:
        raise ValueError("end_ageはstart_age以上である必要があります")
    if terminal_start < 0:
        raise ValueError("terminal_startは0以上である必要があります")
    if terminal_end <= terminal_start:
        raise ValueError("terminal_endはterminal_startより大きい必要があります")
    if growth_ma_periods < 1:
        raise ValueError("growth_ma_periodsは1以上である必要があります")
    
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
    
    # 年齢データの制約チェック
    if np.any(ages < 0):
        raise ValueError("年齢データは0以上である必要があります")
    
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
        
        # 成長率を-1より大きく1以下の範囲に補正
        growth_rate = np.clip(growth_rate, -0.999999, 1.0)
        
        result_df['growth_rate'] = growth_rate
        
        # 移動平均の計算と追加
        if include_growth_ma:
            # 各時点での移動平均を計算
            growth_ma_values = []
            
            for i in range(len(result_df)):
                # 現在の時点を中心とした期間を計算
                # 例：3期間の場合、29歳なら28,29,30歳（i-1, i, i+1）
                center_offset = (growth_ma_periods - 1) // 2
                start_idx = max(0, i - center_offset)
                end_idx = min(len(result_df), i + growth_ma_periods - center_offset)
                
                # 初期時点の場合は利用可能なデータから期間分を取得
                if i < center_offset:
                    start_idx = 0
                    end_idx = min(len(result_df), growth_ma_periods)
                
                # 利用可能な成長率データを取得（NaNを除外）
                available_growth = result_df['growth_rate'].iloc[start_idx:end_idx]
                valid_growth = available_growth.dropna()
                
                # 有効なデータがある場合は平均を計算、なければNaN
                if len(valid_growth) > 0:
                    ma_value = valid_growth.mean()
                else:
                    ma_value = np.nan
                
                growth_ma_values.append(ma_value)
            
            result_df['growth_rate_ma'] = growth_ma_values
    
    return result_df