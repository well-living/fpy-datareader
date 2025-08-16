import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from jpy_datareader.estat import StatsDataReader


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

def calculate_age_midpoint(age_class_str, method='discrete', age_class_width=5):
    """
    年齢階級の文字列から平均値を計算する関数
    
    Parameters
    -------
    age_class_str (str): 年齢階級を表す文字列
    method (str): 計算方法
        - 'discrete': 離散的解釈 (30歳～34歳) → (30+34)/2 = 32.0
        - 'continuous': 連続的解釈 (30歳0か月～34歳12か月) → (30+35)/2 = 32.5
    age_class_width (int): 階級幅が明記されていない場合のデフォルト値
    
    Returns
    -------
    float: 年齢階級の中央値
    """
    # 全角数字を半角に変換
    normalized_str = age_class_str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    numbers = extract_numbers(normalized_str)
    
    # 階級幅を抽出（5歳階級、10歳階級など）
    class_width = age_class_width  # デフォルト値を設定
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

def add_age_midpoint_column(df, age_column_name, income_column_name, method='discrete', age_class_width=5):
    """
    データフレームに年齢階級の中央値列を追加する関数
    
    Parameters
    -------
    df (pd.DataFrame): 対象のデータフレーム
    age_column_name (str): 年齢階級が入っている列の名前
    income_column_name (str): 所得が入っている列の名前
    method (str): 計算方法
        - 'discrete': 離散的解釈 
        - 'continuous': 連続的解釈
    age_class_width (int): 階級幅が明記されていない場合のデフォルト値
    
    Returns
    -------
    pd.DataFrame: age(float)とincomeの2列のデータフレーム
    """
    result_df = pd.DataFrame()
    result_df['age'] = df[age_column_name].apply(
        lambda x: calculate_age_midpoint(x, method, age_class_width)
    )
    result_df['income'] = df[income_column_name]
    return result_df



def interpolate_age_income(
        data, 
        start_age: int=20, 
        end_age: int=65, 
        method: str='linear', 
        extrapolation='constant', 
        terminal_start: int=0, 
        terminal_end: int=100,
        include_growth_rate: bool=False,
        include_growth_ma: bool=False,
        growth_ma_periods:int =3
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


def create_age_income_dataframe_with_midpoint(
        age_data: pd.Series, 
        income_data: pd.Series, 
    ) -> pd.DataFrame:
    """
    年齢階級と収入データから年齢中央値を使ったage, income形式のDataFrameを作成
    Create DataFrame with age midpoints and income columns from age class and income data
    
    Parameters
    ----------
    age_data : pd.Series
        Series containing age class strings (e.g., "30～34歳", "35～39歳")
    income_data : pd.Series
        Series containing income values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['age', 'income'] where:
        - 'age' contains calculated age midpoints (float)
        - 'income' contains income values
    """
    # 一時的なDataFrameを作成
    temp_df = pd.DataFrame({
        'age': age_data,
        'income': income_data
    })
    
    # 既存の関数を使用して年齢中央値を追加
    return add_age_midpoint_column(temp_df, 'age', 'income', method='continuous')


# 家計調査データ変換関数群
def get_family_income_expenditure_survey_data(
        api_key: str,
        cat01_code: str, 
        value_name: str,
        stats_data_id: str = "0002070011", 
        yyyy: int = 2024
    ) -> pd.DataFrame:
    """
    家計調査データを取得
    二人以上の世帯 年次
    Retrieve Family Income and Expenditure Survey data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    cat01_code : str
        Category 01 code for data filtering
    value_name : str
        Name for the value column in returned DataFrame
        "age", "income"
    stats_data_id : str
        Statistics data ID (e.g., '0002070011')
    yyyy : int, default 2024
        Year for data retrieval
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with household head age class and specified values
    """
    params = {
        "cdCat01": cat01_code,
        "cdCat02": "04",  # 二人以上の世帯のうち勤労者世帯 (Households of two or more persons - Worker households)
        "cdTime": f"{yyyy}000000",
    }
    
    data = StatsDataReader(
        api_key=api_key,
        statsDataId=stats_data_id,
        **params
    )
    value = data.read()
    
    # フィルタリング (Filter out average and 65+ categories)
    cols = ["世帯主の年齢階級", "値"]
    df = value.loc[~value["世帯主の年齢階級"].isin(["平均", "65歳以上"]), cols]
    return df.rename(columns={"値": value_name})

def process_family_income_expenditure_survey_data(
        api_key: str, 
        stats_data_id: str = "0002070011", 
        yyyy: int = 2024,
        start_age: int = 15,
        end_age: int = 80,
        method: str = 'linear',
        extrapolation: str = 'terminal',
        terminal_start: int = 0,
        terminal_end: int = 100,
        include_growth_rate: bool = True,
        include_growth_ma: bool = True,
        growth_ma_periods: int = 3
    ) -> pd.DataFrame:
    """
    家計調査データの処理
    Process Family Income and Expenditure Survey data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    stats_data_id : str, default "0002070011"
        Statistics data ID. Common values:
        - '0002070011': 用途分類（世帯主の年齢階級別）年次 (Purpose classification by household head age class, annual)
    yyyy : int, default 2024
        Year for data processing
    start_age : int, default 15
        Starting age for interpolation
    end_age : int, default 80
        Ending age for interpolation
    method : str, default 'linear'
        Interpolation method: 'linear' or 'cubic_spline'
    extrapolation : str, default 'constant'
        Extrapolation method: 'constant' or 'terminal'
    terminal_start : int, default 0
        Terminal start age (when extrapolation='terminal')
    terminal_end : int, default 100
        Terminal end age (when extrapolation='terminal')
    include_growth_rate : bool, default True
        Whether to include growth rate column
    include_growth_ma : bool, default True
        Whether to include growth rate moving average
    growth_ma_periods : int, default 3
        Moving average periods
    
    Returns
    -------
    pd.DataFrame
        Interpolated age-income table with columns:
        - 'age': Integer ages from start_age to end_age
        - 'income': Interpolated disposable income values (yen)
        - 'growth_rate': Year-over-year growth rate
        - 'growth_rate_ma': 3-period moving average of growth rate
    """
    # 年齢と可処分所得データの取得 (Get age and disposable income data)
    age_df = get_family_income_expenditure_survey_data(api_key, "009", "世帯主の年齢", stats_data_id, yyyy)
    income_df = get_family_income_expenditure_survey_data(api_key, "233", "可処分所得", stats_data_id, yyyy)
    
    # データのマージ (Merge data)
    merged_df = age_df.merge(income_df, on="世帯主の年齢階級")
    
    # interpolate_age_income関数に渡すためのデータ形式に変換
    data_for_interpolation = merged_df[["世帯主の年齢", "可処分所得"]]
    
    # 補間処理
    return interpolate_age_income(
        data_for_interpolation,
        start_age=start_age,
        end_age=end_age,
        method=method,
        extrapolation=extrapolation,
        terminal_start=terminal_start,
        terminal_end=terminal_end,
        include_growth_rate=include_growth_rate,
        include_growth_ma=include_growth_ma,
        growth_ma_periods=growth_ma_periods
    )


# 全国家計構造調査データ変換関数群
def get_family_income_consumption_wealth_survey_data(
        api_key: str, 
        cat04_code: str, 
        value_name: str,
        stats_data_id: str = '0003424621'
    ) -> pd.DataFrame:
    """
    全国家計構造調査データを取得
    Retrieve National Survey of Family Income, Consumption and Wealth data
    二人以上の世帯
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    cat04_code : str
        Category 04 code for data filtering
    value_name : str
        Name for the value column in returned DataFrame
    stats_data_id : str
        Statistics data ID (e.g., '0003424621')
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing 5-year age class data
    """
    params = {
        "cdCat01": "1",  # 二人以上の世帯 (Households of two or more persons)
        "cdCat02": "1",  # 勤労者世帯 (Worker households)
        "cdCat03": "0",  # 性別 平均 (Average)
        "cdCat04": cat04_code,
    }
    
    data = StatsDataReader(
        api_key=api_key,
        statsDataId=stats_data_id,
        **params
    )
    value = data.read()
    
    # 5歳階級のデータのみを抽出 (Extract only 5-year age class data)
    cond = value["世帯主の年齢階級32区分"].str.contains("５歳階級")
    df = value.loc[cond, ["世帯主の年齢階級32区分", "値"]]
    return df.rename(columns={"値": value_name})

def process_family_income_consumption_wealth_survey_data(
        api_key: str, 
        stats_data_id: str = "0003424621", 
        start_age: int = 15,
        end_age: int = 80,
        method: str = 'linear',
        extrapolation: str = 'terminal',
        terminal_start: int = 0,
        terminal_end: int = 100,
        include_growth_rate: bool = True,
        include_growth_ma: bool = True,
        growth_ma_periods: int = 3
    ) -> pd.DataFrame:
    """
    全国家計構造調査データの処理
    Process National Survey of Family Income, Consumption and Wealth data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    stats_data_id : str
        Statistics data ID. Common values:
        - '0003424621': 1世帯当たり1か月間の収入と支出 (Monthly income and expenditure per household)
        - '0003426498': 世帯の種類,世帯主の年齢階級,所得構成別1世帯当たり年間収入額 (Annual income by household type, age class, and income composition)
    start_age : int, default 15
        Starting age for interpolation
    end_age : int, default 80
        Ending age for interpolation
    method : str, default 'linear'
        Interpolation method: 'linear' or 'cubic_spline'
    extrapolation : str, default 'terminal'
        Extrapolation method: 'constant' or 'terminal'
    terminal_start : int, default 0
        Terminal start age (when extrapolation='terminal')
    terminal_end : int, default 100
        Terminal end age (when extrapolation='terminal')
    include_growth_rate : bool, default True
        Whether to include growth rate column
    include_growth_ma : bool, default True
        Whether to include growth rate moving average
    growth_ma_periods : int, default 3
        Moving average periods
    
    Returns
    -------
    pd.DataFrame
        Interpolated age-income table with columns:
        - 'age': Integer ages from start_age to end_age
        - 'income': Interpolated disposable income values (yen)
        - 'growth_rate': Year-over-year growth rate
        - 'growth_rate_ma': 3-period moving average of growth rate
    """
    # 可処分所得データの取得 (Get disposable income data)
    income_df = get_family_income_consumption_wealth_survey_data(api_key, "4", "可処分所得", stats_data_id)
    
    # age, income形式のDataFrameを作成
    # 年齢の列がないため、中央値補完が必要
    age_income_df = create_age_income_dataframe_with_midpoint( 
        income_df["世帯主の年齢階級32区分"],
        income_df["可処分所得"],
    )
    
    # 補間処理（既存関数を活用）
    return interpolate_age_income(
        age_income_df,
        start_age=start_age,
        end_age=end_age,
        method=method,
        extrapolation=extrapolation,
        terminal_start=terminal_start,
        terminal_end=terminal_end,
        include_growth_rate=include_growth_rate,
        include_growth_ma=include_growth_ma,
        growth_ma_periods=growth_ma_periods
    )


# 賃金構造基本調査データ変換関数群
def get_wage_structure_survey_data_by_tab(
        api_key: str, 
        value_name: str, 
        cd_tab: str,
        stats_data_id: str = "0003425893",
        cd_cat01: str = "01", 
        cd_cat02: str = "01", 
        cd_cat03: str = "01",
        cd_cat05: str = "01", 
        cd_cat06: str = "01", 
        yyyy: int = 2023
    ) -> pd.DataFrame:
    """
    賃金構造基本調査データを取得
    Retrieve Basic Survey on Wage Structure data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    value_name : str
        Name for the value column in returned DataFrame
    cd_tab : str
        Table category code (e.g., '33' for age, '40' for monthly salary, '44' for bonus)
    stats_data_id : str
        Statistics data ID (e.g., '0003425893')
    cd_cat01 : str, default "01" 
        Enterprise scale category '企業規模_基本' 企業規模計（10人以上）
    cd_cat02 : str, default "01" 
        Industry classification category
    cd_cat03 : str, default "01" 産業分類	 Ｔ２ 産業計
        Gender category  '性別_基本', 男女計
    cd_cat05 : str, default "01"
        Education category 学歴_基本８区分（2020年～） 学歴計
    cd_cat06 : str, default "01"
        Public/private sector category  民・公区分 民営＋公営
    yyyy : int, default 2023
        Year for data retrieval
    
    Returns
    -------
    pd.DataFrame
        DataFrame with age class and specified values, excluding total categories
    """
    params = {
        "cdTab": cd_tab,
        "cdCat01": cd_cat01,
        "cdCat02": cd_cat02,
        "cdCat03": cd_cat03,
        "cdCat05": cd_cat05,
        "cdCat06": cd_cat06,
        "cdTime": f"{yyyy}000000"
    }
    
    data = StatsDataReader(
        api_key=api_key,
        statsDataId=stats_data_id,
        **params
    )
    value = data.read()
    
    # 年齢計以外のデータを抽出 (Extract data excluding age total)
    df = value.loc[~value["年齢階級_基本"].isin(["年齢計"]), ['年齢階級_基本', '値']]
    return df.rename(columns={"値": value_name})

def process_wage_structure_survey_data(
        api_key: str, 
        stats_data_id: str = '0003425893', 
        yyyy: int = 2023,
        start_age: int = 15,
        end_age: int = 80,
        method: str = 'linear',
        extrapolation: str = 'terminal',
        terminal_start: int = 0,
        terminal_end: int = 100,
        include_growth_rate: bool = True,
        include_growth_ma: bool = True,
        growth_ma_periods: int = 3
    ) -> pd.DataFrame:
    """
    賃金構造基本調査データの処理
    Process Basic Survey on Wage Structure data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    stats_data_id : str
        Statistics data ID. Common values:
        - '0003425893': 一般_産業大・中分類_年齢階級別DB (General - by major/medium industry classification and age class)
    yyyy : int, default 2023
        Year for data processing
    start_age : int, default 15
        Starting age for interpolation
    end_age : int, default 80
        Ending age for interpolation
    method : str, default 'linear'
        Interpolation method: 'linear' or 'cubic_spline'
    extrapolation : str, default 'terminal'
        Extrapolation method: 'constant' or 'terminal'
    terminal_start : int, default 0
        Terminal start age (when extrapolation='terminal')
    terminal_end : int, default 100
        Terminal end age (when extrapolation='terminal')
    include_growth_rate : bool, default True
        Whether to include growth rate column
    include_growth_ma : bool, default True
        Whether to include growth rate moving average
    growth_ma_periods : int, default 3
        Moving average periods
    
    Returns
    -------
    pd.DataFrame
        Interpolated age-income table with columns:
        - 'age': Integer ages from start_age to end_age
        - 'income': Interpolated annual income values (yen, calculated as monthly*12 + bonus)
        - 'growth_rate': Year-over-year growth rate
        - 'growth_rate_ma': 3-period moving average of growth rate
    """
    # 年齢、月給、賞与データの取得 (Get age, monthly salary, and bonus data)
    age_df = get_wage_structure_survey_data_by_tab(api_key, "age", "33", stats_data_id, yyyy=yyyy) # 年齢はfloat
    salary_df = get_wage_structure_survey_data_by_tab(api_key, "salary", "40", stats_data_id, yyyy=yyyy)
    bonus_df = get_wage_structure_survey_data_by_tab(api_key, "bonus", "44", stats_data_id, yyyy=yyyy)
    
    # 年収計算（月給×12 + 賞与）× 1000（千円→円）
    # Calculate annual income (monthly*12 + bonus) * 1000 (thousand yen to yen)
    combined_df = age_df.copy()
    combined_df["income"] = (salary_df["salary"] * 12 + bonus_df["bonus"]) * 1000
    
    # age, income形式のDataFrameを作成
    # Create age-income DataFrame (actual age values, not age classes)
    age_income_df = pd.DataFrame({
        'age': combined_df["age"],
        'income': combined_df["income"]
    })
    
    # 補間処理（既存関数を活用）
    return interpolate_age_income(
        age_income_df,
        start_age=start_age,
        end_age=end_age,
        method=method,
        extrapolation=extrapolation,
        terminal_start=terminal_start,
        terminal_end=terminal_end,
        include_growth_rate=include_growth_rate,
        include_growth_ma=include_growth_ma,
        growth_ma_periods=growth_ma_periods
    )


# 賃金構造基本調査（都道府県別）データ変換関数群
def get_wage_structure_survey_data_by_prefecture(
        api_key: str, 
        value_name: str, 
        cd_tab: str,
        stats_data_id: str = "0003426933",
        cd_cat03: str = "01", 
        cd_cat04: str = "01", 
        area: str = "13000", 
        yyyy: int = 2023
    ) -> pd.DataFrame:
    """
    賃金構造基本調査（都道府県別）データを取得
    Retrieve Basic Survey on Wage Structure data by prefecture
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    value_name : str
        Name for the value column in returned DataFrame
    cd_tab : str
        Table category code (e.g., '33' for age, '40' for monthly salary, '44' for bonus)
    stats_data_id : str
        Statistics data ID (e.g., '0003426933')
    cd_cat03 : str, default "01"
        Enterprise scale category 企業規模_基本 企業規模計（10人以上）
    cd_cat04 : str, default "01"
        Industry classification category '産業分類', Ｔ２ 産業計
    area : str, default "13000"
        Area code (e.g., "13000" for Tokyo, "27000" for Osaka)
    yyyy : int, default 2023
        Year for data retrieval
    
    Returns
    -------
    pd.DataFrame
        DataFrame with age class and specified values for the specified prefecture
    """
    params = {
        "cdTab": cd_tab,
        "cdCat01": "01",  # 性別_基本' 男女計 (Both sexes)
        "cdCat03": cd_cat03,
        "cdCat04": cd_cat04,
        "cdArea": area,
        "cdTime": f"{yyyy}000000"
    }
    
    data = StatsDataReader(
        api_key=api_key,
        statsDataId=stats_data_id,
        **params
    )
    value = data.read()
    
    # 年齢計以外のデータを抽出 (Extract data excluding age total)
    df = value.loc[~value["年齢階級_基本"].isin(["年齢計"]), ['年齢階級_基本', '値']]
    return df.rename(columns={"値": value_name})


def process_wage_structure_survey_data_by_prefecture(
        api_key: str, 
        stats_data_id: str = "0003426933",
        area: str = "13000", 
        yyyy: int = 2023,
        start_age: int = 15,
        end_age: int = 80,
        method: str = 'linear',
        extrapolation: str = 'terminal',
        terminal_start: int = 0,
        terminal_end: int = 100,
        include_growth_rate: bool = True,
        include_growth_ma: bool = True,
        growth_ma_periods: int = 3
    ) -> pd.DataFrame:
    """
    賃金構造基本調査（都道府県別）データの処理
    Process Basic Survey on Wage Structure data by prefecture
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    stats_data_id : str
        Statistics data ID. Common values:
        - '0003426933': 一般_都道府県別_年齢階級別DB (General - by prefecture and age class)
    area : str, default "13000"
        Area code (e.g., "13000" for Tokyo, "27000" for Osaka, "01000" for Hokkaido)
    yyyy : int, default 2023
        Year for data processing
    start_age : int, default 15
        Starting age for interpolation
    end_age : int, default 80
        Ending age for interpolation
    method : str, default 'linear'
        Interpolation method: 'linear' or 'cubic_spline'
    extrapolation : str, default 'terminal'
        Extrapolation method: 'constant' or 'terminal'
    terminal_start : int, default 0
        Terminal start age (when extrapolation='terminal')
    terminal_end : int, default 100
        Terminal end age (when extrapolation='terminal')
    include_growth_rate : bool, default True
        Whether to include growth rate column
    include_growth_ma : bool, default True
        Whether to include growth rate moving average
    growth_ma_periods : int, default 3
        Moving average periods
    
    Returns
    -------
    pd.DataFrame
        Interpolated age-income table with columns:
        - 'age': Integer ages from start_age to end_age
        - 'income': Interpolated annual income values (yen, calculated as monthly*12 + bonus)
        - 'growth_rate': Year-over-year growth rate
        - 'growth_rate_ma': 3-period moving average of growth rate
    """
    # 年齢、月給、賞与データの取得 (Get age, monthly salary, and bonus data)
    age_df = get_wage_structure_survey_data_by_prefecture(api_key, "age", "33", stats_data_id, area=area, yyyy=yyyy)
    salary_df = get_wage_structure_survey_data_by_prefecture(api_key, "salary", "40", stats_data_id, area=area, yyyy=yyyy)
    bonus_df = get_wage_structure_survey_data_by_prefecture(api_key, "bonus", "44", stats_data_id, area=area, yyyy=yyyy)
    
    # 年収計算（月給×12 + 賞与）× 1000（千円→円）
    # Calculate annual income (monthly*12 + bonus) * 1000 (thousand yen to yen)
    combined_df = age_df.copy()
    combined_df["income"] = (salary_df["salary"] * 12 + bonus_df["bonus"]) * 1000
    
    # age, income形式のDataFrameを作成
    # Create age-income DataFrame (actual age values, not age classes)
    age_income_df = pd.DataFrame({
        'age': combined_df["age"],
        'income': combined_df["income"]
    })
    
    # 補間処理（既存関数を活用）
    return interpolate_age_income(
        age_income_df,
        start_age=start_age,
        end_age=end_age,
        method=method,
        extrapolation=extrapolation,
        terminal_start=terminal_start,
        terminal_end=terminal_end,
        include_growth_rate=include_growth_rate,
        include_growth_ma=include_growth_ma,
        growth_ma_periods=growth_ma_periods
    )


# 国民生活基礎調査データ変換関数群
def get_living_conditions_survey_data(
        api_key: str, 
        value_name: str,
        stats_data_id: str = "0004021242", 
        cd_tab: str = "10",
        cd_cat02: str = "038"
    ) -> pd.DataFrame:
    """
    国民生活基礎調査データを取得
    Retrieve Comprehensive Survey of Living Conditions data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    value_name : str
        Name for the value column in returned DataFrame
    stats_data_id : str
        Statistics data ID (e.g., '0004021242')
    cd_tab : str, default "10"
        Table category code ("10" for average income per household)　１世帯当たり平均所得金額
    cd_cat02 : str, default "038"
        Category 02 code ("038" for 2022 data)　2022(令和4)年
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 10-year age class data, excluding totals and sub-totals
    """
    params = {
        "cdTab": cd_tab,
        "cdCat02": cd_cat02,
    }
    
    data = StatsDataReader(
        api_key=api_key,
        statsDataId=stats_data_id,
        **params
    )
    value = data.read()
    
    # 総数、再掲以外のデータを抽出 (Extract data excluding totals and sub-totals)
    cond = ~value["世帯主の年齢（１０歳階級）_102"].str.contains("総数|再掲")
    df = value.loc[cond, ['世帯主の年齢（１０歳階級）_102', '値']]
    return df.rename(columns={"値": value_name})


def process_living_conditions_survey_data(
        api_key: str, 
        stats_data_id: str = "0004021242", 
        start_age: int = 15,
        end_age: int = 80,
        method: str = 'linear',
        extrapolation: str = 'terminal',
        terminal_start: int = 0,
        terminal_end: int = 100,
        include_growth_rate: bool = True,
        include_growth_ma: bool = True,
        growth_ma_periods: int = 3
    ) -> pd.DataFrame:
    """
    国民生活基礎調査データの処理
    Process Comprehensive Survey of Living Conditions data
    
    Parameters
    ----------
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    stats_data_id : str
        Statistics data ID. Common values:
        - '0004021242': １世帯当たり平均所得金額,世帯人員１人当たり平均所得金額,世帯主の年齢（10歳階級）・年次別
          (Average income per household, per person, by household head age (10-year class) and year)
    start_age : int, default 15
        Starting age for interpolation
    end_age : int, default 80
        Ending age for interpolation
    method : str, default 'linear'
        Interpolation method: 'linear' or 'cubic_spline'
    extrapolation : str, default 'terminal'
        Extrapolation method: 'constant' or 'terminal'
    terminal_start : int, default 0
        Terminal start age (when extrapolation='terminal')
    terminal_end : int, default 100
        Terminal end age (when extrapolation='terminal')
    include_growth_rate : bool, default True
        Whether to include growth rate column
    include_growth_ma : bool, default True
        Whether to include growth rate moving average
    growth_ma_periods : int, default 3
        Moving average periods
    
    Returns
    -------
    pd.DataFrame
        Interpolated age-income table with columns:
        - 'age': Integer ages from start_age to end_age
        - 'income': Interpolated income values (yen, converted from 10,000 yen units)
        - 'growth_rate': Year-over-year growth rate
        - 'growth_rate_ma': 3-period moving average of growth rate
    """
    # 収入データの取得
    income_df = get_living_conditions_survey_data(api_key, "income", stats_data_id)
    
    # age, income形式のDataFrameを作成
    # Create age-income DataFrame using existing function
    age_income_df = create_age_income_dataframe_with_midpoint( # 年齢がfloatではないため補完が必要
        income_df["世帯主の年齢（１０歳階級）_102"],
        income_df["income"] * 10000,  # 万円を円に変換 (Convert from 10,000 yen to yen)
        "世帯主の年齢（１０歳階級）_102"
    )
    
    # 補間処理（既存関数を活用）
    return interpolate_age_income(
        age_income_df,
        start_age=start_age,
        end_age=end_age,
        method=method,
        extrapolation=extrapolation,
        terminal_start=terminal_start,
        terminal_end=terminal_end,
        include_growth_rate=include_growth_rate,
        include_growth_ma=include_growth_ma,
        growth_ma_periods=growth_ma_periods
    )


def create_age_income_table(data_source: str, api_key: str, **kwargs) -> pd.DataFrame:
    """
    データソースに応じた年齢-収入テーブルを作成
    Create age-income table based on specified data source
    
    Parameters
    ----------
    data_source : str
        Data source name. Supported values:
        - 'family_income_expenditure': Family Income and Expenditure Survey
        - 'family_income_consumption_wealth': National Survey of Family Income, Consumption and Wealth
        - 'wage_structure': Basic Survey on Wage Structure
        - 'wage_structure_prefecture': Basic Survey on Wage Structure (by Prefecture)
        - 'living_conditions': Comprehensive Survey of Living Conditions
    api_key : str
        e-Stat API key for accessing Japanese government statistics
    **kwargs
        Additional parameters specific to each converter:
        
        For 'family_income_expenditure':
            - stats_data_id (str): Statistics data ID
            - yyyy (int): Year for data retrieval
            - start_age (int): Starting age for interpolation
            - end_age (int): Ending age for interpolation
            - method (str): Interpolation method
            - extrapolation (str): Extrapolation method
            - terminal_start (int): Terminal start age
            - terminal_end (int): Terminal end age
            - include_growth_rate (bool): Include growth rate column
            - include_growth_ma (bool): Include growth rate moving average
            - growth_ma_periods (int): Moving average periods
            
        For 'family_income_consumption_wealth':
            - stats_data_id (str): Statistics data ID
            - start_age (int): Starting age for interpolation
            - end_age (int): Ending age for interpolation
            - method (str): Interpolation method
            - extrapolation (str): Extrapolation method
            - terminal_start (int): Terminal start age
            - terminal_end (int): Terminal end age
            - include_growth_rate (bool): Include growth rate column
            - include_growth_ma (bool): Include growth rate moving average
            - growth_ma_periods (int): Moving average periods
            
        For 'wage_structure':
            - stats_data_id (str): Statistics data ID
            - yyyy (int): Year for data retrieval
            - start_age (int): Starting age for interpolation
            - end_age (int): Ending age for interpolation
            - method (str): Interpolation method
            - extrapolation (str): Extrapolation method
            - terminal_start (int): Terminal start age
            - terminal_end (int): Terminal end age
            - include_growth_rate (bool): Include growth rate column
            - include_growth_ma (bool): Include growth rate moving average
            - growth_ma_periods (int): Moving average periods
            
        For 'wage_structure_prefecture':
            - stats_data_id (str): Statistics data ID
            - area (str): Area code (e.g., '13000' for Tokyo)
            - yyyy (int): Year for data retrieval
            - start_age (int): Starting age for interpolation
            - end_age (int): Ending age for interpolation
            - method (str): Interpolation method
            - extrapolation (str): Extrapolation method
            - terminal_start (int): Terminal start age
            - terminal_end (int): Terminal end age
            - include_growth_rate (bool): Include growth rate column
            - include_growth_ma (bool): Include growth rate moving average
            - growth_ma_periods (int): Moving average periods
            
        For 'living_conditions':
            - stats_data_id (str): Statistics data ID
            - start_age (int): Starting age for interpolation
            - end_age (int): Ending age for interpolation
            - method (str): Interpolation method
            - extrapolation (str): Extrapolation method
            - terminal_start (int): Terminal start age
            - terminal_end (int): Terminal end age
            - include_growth_rate (bool): Include growth rate column
            - include_growth_ma (bool): Include growth rate moving average
            - growth_ma_periods (int): Moving average periods
    
    Returns
    -------
    pd.DataFrame
        Interpolated age-income table with columns:
        - 'age': Integer ages from start_age to end_age
        - 'income': Interpolated income values (yen)
        - 'growth_rate': Year-over-year growth rate
        - 'growth_rate_ma': 3-period moving average of growth rate
    
    Raises
    ------
    ValueError
        If an unsupported data_source is specified
    KeyError
        If required parameters are missing
    
    Examples
    --------
    >>> # Family Income and Expenditure Survey data
    >>> family_income_table = create_age_income_table(
    ...     'family_income_expenditure', 
    ...     api_key, 
    ...     stats_data_id='0002070011', 
    ...     yyyy=2024,
    ...     start_age=20,
    ...     end_age=65,
    ...     method='linear',
    ...     extrapolation='constant',
    ...     terminal_start=0,
    ...     terminal_end=100,
    ...     include_growth_rate=True,
    ...     include_growth_ma=True,
    ...     growth_ma_periods=3
    ... )
    
    >>> # Basic Survey on Wage Structure for Tokyo
    >>> tokyo_wages = create_age_income_table(
    ...     'wage_structure_prefecture', 
    ...     api_key, 
    ...     stats_data_id='0003426933',
    ...     area='13000',
    ...     yyyy=2023,
    ...     start_age=20,
    ...     end_age=65,
    ...     method='linear',
    ...     extrapolation='constant',
    ...     terminal_start=0,
    ...     terminal_end=100,
    ...     include_growth_rate=True,
    ...     include_growth_ma=True,
    ...     growth_ma_periods=3
    ... )
    """
    if data_source == 'family_income_expenditure':
        return process_family_income_expenditure_survey_data(
            api_key,
            kwargs.get('stats_data_id'),
            kwargs.get('yyyy'),
            kwargs.get('start_age', 15),
            kwargs.get('end_age', 80),
            kwargs.get('method', 'linear'),
            kwargs.get('extrapolation', 'terminal'),
            kwargs.get('terminal_start', 0),
            kwargs.get('terminal_end', 100),
            kwargs.get('include_growth_rate', True),
            kwargs.get('include_growth_ma', True),
            kwargs.get('growth_ma_periods', 3)
        )
    
    elif data_source == 'family_income_consumption_wealth':
        return process_family_income_consumption_wealth_survey_data(
            api_key,
            kwargs.get('stats_data_id'),
            kwargs.get('start_age', 15),
            kwargs.get('end_age', 80),
            kwargs.get('method', 'linear'),
            kwargs.get('extrapolation', 'terminal'),
            kwargs.get('terminal_start', 0),
            kwargs.get('terminal_end', 100),
            kwargs.get('include_growth_rate', True),
            kwargs.get('include_growth_ma', True),
            kwargs.get('growth_ma_periods', 3)
        )
    
    elif data_source == 'wage_structure':
        return process_wage_structure_survey_data(
            api_key,
            kwargs.get('stats_data_id'),
            kwargs.get('yyyy'),
            kwargs.get('start_age', 15),
            kwargs.get('end_age', 80),
            kwargs.get('method', 'linear'),
            kwargs.get('extrapolation', 'terminal'),
            kwargs.get('terminal_start', 0),
            kwargs.get('terminal_end', 100),
            kwargs.get('include_growth_rate', True),
            kwargs.get('include_growth_ma', True),
            kwargs.get('growth_ma_periods', 3)
        )
    
    elif data_source == 'wage_structure_prefecture':
        return process_wage_structure_survey_data_by_prefecture(
            api_key,
            kwargs.get('stats_data_id'),
            kwargs.get('area'),
            kwargs.get('yyyy'),
            kwargs.get('start_age', 15),
            kwargs.get('end_age', 80),
            kwargs.get('method', 'linear'),
            kwargs.get('extrapolation', 'terminal'),
            kwargs.get('terminal_start', 0),
            kwargs.get('terminal_end', 100),
            kwargs.get('include_growth_rate', True),
            kwargs.get('include_growth_ma', True),
            kwargs.get('growth_ma_periods', 3)
        )
    
    elif data_source == 'living_conditions':
        return process_living_conditions_survey_data(
            api_key,
            kwargs.get('stats_data_id'),
            kwargs.get('start_age', 15),
            kwargs.get('end_age', 80),
            kwargs.get('method', 'linear'),
            kwargs.get('extrapolation', 'terminal'),
            kwargs.get('terminal_start', 0),
            kwargs.get('terminal_end', 100),
            kwargs.get('include_growth_rate', True),
            kwargs.get('include_growth_ma', True),
            kwargs.get('growth_ma_periods', 3)
        )
    
    else:
        raise ValueError(f"サポートされていないデータソース: {data_source}")