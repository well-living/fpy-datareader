# fpy_datareader/child_educational_expenses.py
"""
文部科学省 子供の学習費調査データの取得・処理モジュール

このモジュールは文部科学省の子供の学習費調査データを取得し、
分析しやすい形式に変換するための機能を提供します。

データソース:
- 文部科学省 子供の学習費調査
  https://www.mext.go.jp/b_menu/toukei/chousa03/gakushuuhi/1268091.htm
- e-Stat 令和５年度 子供の学習費調査
  https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00400201&tstat=000001012023&cycle=0&tclass1=000001224200&tclass2=000001224201&tclass3=000001224327&tclass4val=0

メモ:
  openpyxlのインストールが必要
"""

import re
from io import StringIO
from typing import Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd


# ================================
# 設定定数
# ================================

# データソースURL設定
TUITION_BASE_URL = "https://www.e-stat.go.jp/stat-search/file-download"

TUITION_STAT_IDS = {
    'kindergarten': '000040233660',
    'elementary': '000040233661', 
    'middle': '000040233662',
    'high': '000040233663'
}

# Excel読み込み設定
TUITION_SKIPROWS = {
    'kindergarten': 10,
    'elementary': 10,
    'middle': 10,
    'high': 9
}

TUITION_AGE_OR_GRADE = {
    'kindergarten': '年齢別',
    'elementary': '学年別',
    'middle': '学年別', 
    'high': '学年別'
}

# 欠損値表記
TUITION_NA_VALUES = ["…", "－"]

# カラム名
TUITION_TIDY_COLUMNS = ["学習費区分", "学校", "学年", "学習費"]

# 年齢計算オフセット
TUITION_AGE_OFFSETS = {
    'kindergarten': 2,
    'elementary': 5,
    'middle': 11,
    'high': 14
}

# ファイルパス（dataディレクトリ内を想定）
TUITION_MAPPING_FILE = Path(__file__).parent / "data" / "tuition_mapping.csv"
# パッケージ直下に置く場合は以下を使用
# TUITION_MAPPING_FILE = Path(__file__).parent / "tuition_mapping.csv"


# ================================
# ユーティリティ関数
# ================================

def extract_tuition_grade_info(grade_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    学年文字列から年齢と学年情報を抽出
    
    Parameters
    ----------
    grade_str : str
        学年情報を含む文字列（例：「3歳」、「第1学年」）
        
    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        (年齢, 学年)のタプル。幼稚園の場合は(年齢, None)、
        学校の場合は(None, 学年)、パターンに一致しない場合は(None, None)
        
    Examples
    --------
    >>> extract_tuition_grade_info("3歳")
    (3, None)
    >>> extract_tuition_grade_info("第1学年")  
    (None, 1)
    >>> extract_tuition_grade_info("不明")
    (None, None)
    """
    if "歳" in grade_str:
        match = re.search(r'(\d+)歳', grade_str)
        return int(match.group(1)) if match else None, None
    elif "学年" in grade_str:
        match = re.search(r'第(\d+)学年', grade_str)
        return None, int(match.group(1)) if match else None
    return None, None


def calculate_tuition_age_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    ベクトル化演算を使用した学習費データの年齢計算
    
    Parameters
    ----------
    df : pd.DataFrame
        '学校'と'学年'列を含むDataFrame
        
    Returns
    -------
    pd.Series
        計算された年齢のSeries。教育段階別の年齢オフセットを使用
        
    Examples
    --------
    >>> data = {'学校': ['幼稚園', '小学校', '中学校', '高等学校'], '学年': [1, 1, 1, 1]}
    >>> df = pd.DataFrame(data)
    >>> ages = calculate_tuition_age_vectorized(df)
    >>> ages.tolist()
    [3, 6, 12, 15]
    """
    conditions = [
        df["学校"].str.contains("幼稚園"),
        df["学校"].str.contains("小学校"),
        df["学校"].str.contains("中学校"),
        df["学校"].str.contains("高等学校")
    ]
    
    choices = [
        df["学年"] + TUITION_AGE_OFFSETS['kindergarten'],
        df["学年"] + TUITION_AGE_OFFSETS['elementary'],
        df["学年"] + TUITION_AGE_OFFSETS['middle'],
        df["学年"] + TUITION_AGE_OFFSETS['high']
    ]
    
    return np.select(conditions, choices, default=np.nan)


def update_tuition_index_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    学習費データのインデックスラベルを更新（「その他」の曖昧性解消）
    
    Parameters
    ----------
    df : pd.DataFrame
        インデックスラベルを更新する必要があるDataFrame
        
    Returns
    -------
    pd.DataFrame
        コンテキスト追跡手法を使用して更新されたインデックスラベルを持つDataFrame
        
    Examples
    --------
    >>> df = pd.DataFrame({'値': [1, 2, 3, 4, 5]}, 
    ...                   index=['学校教育費', 'その他', '補助学習費', 'その他', '学校給食費'])
    >>> updated_df = update_tuition_index_labels(df)
    >>> list(updated_df.index)
    ['学校教育費', 'その他学校教育費', '補助学習費', 'その他補助学習費', '学校給食費']
    """
    df_copy = df.copy()
    
    # 「その他」の曖昧性解消のための現在のコンテキストを追跡
    new_index = []
    context_tracker = None
    
    for label in df_copy.index:
        # 「その他」の曖昧性解消のためのコンテキスト追跡
        if label in ["学校教育費", "補助学習費"]:
            context_tracker = label
        
        # 現在のコンテキストに基づいて「その他」を置換
        if label == "その他" and context_tracker:
            new_index.append(f"{label}{context_tracker}")
        else:
            new_index.append(label)
    
    df_copy.index = new_index
    return df_copy


def split_tuition_school_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    学習費データの学校情報を公立・私立と学校種別に分割
    
    Parameters
    ----------
    df : pd.DataFrame
        結合された学校情報を含む'学校'列を持つDataFrame
        
    Returns
    -------
    pd.DataFrame
        分離された学校情報を持つDataFrame:
        - '公立・私立区分': 公立または私立の分類
        - '学校': 公立・私立接頭辞なしの学校種別
        
    Examples
    --------
    >>> df = pd.DataFrame({'学校': ['公立幼稚園', '私立小学校'], '学習費': [100, 200]})
    >>> result = split_tuition_school_info(df)
    >>> result['公立・私立区分'].tolist()
    ['公立', '私立'] 
    >>> result['学校'].tolist()
    ['幼稚園', '小学校']
    """
    df_copy = df.copy()
    
    # 公立・私立と学校種別を抽出
    split_df = df_copy["学校"].str.extract(r'^(公立|私立)(.+)$')
    split_df.columns = ["公立・私立区分", "学校"]
    
    # 元の学校列を分割された情報で置換
    df_result = df_copy.drop(columns="学校").join(split_df)
    
    return df_result


# ================================
# メイン処理関数
# ================================

def get_tuition_urls() -> Dict[str, str]:
    """
    学習費調査データのURLを取得
    
    Returns
    -------
    Dict[str, str]
        教育段階別のURL辞書
        
    Examples
    --------
    >>> urls = get_tuition_urls()
    >>> 'kindergarten' in urls
    True
    >>> len(urls) == 4
    True
    """
    return {
        level: f"{TUITION_BASE_URL}?statInfId={stat_id}&fileKind=0"
        for level, stat_id in TUITION_STAT_IDS.items()
    }


def load_tuition_mapping() -> pd.DataFrame:
    """
    学習費カテゴリマッピングテーブルを読み込み
    
    Returns
    -------
    pd.DataFrame
        カテゴリ階層マッピングテーブル
        
    Examples
    --------
    >>> mapping = load_tuition_mapping()
    >>> mapping.columns.tolist()
    ['level1', 'level2', 'level3', 'level4']
    >>> len(mapping) == 29
    True
    """
    try:
        if TUITION_MAPPING_FILE.exists():
            mapping = pd.read_csv(TUITION_MAPPING_FILE)
        else:
            # フォールバック: 内蔵CSVデータ
            csv_data = """level1,level2,level3,level4
学習費総額,学校教育費,学校教育費,入学金・入園料
学習費総額,学校教育費,学校教育費,入学時に納付した施設整備費等
学習費総額,学校教育費,学校教育費,入学検定料
学習費総額,学校教育費,学校教育費,授業料
学習費総額,学校教育費,学校教育費,施設整備費等
学習費総額,学校教育費,学校教育費,修学旅行費
学習費総額,学校教育費,学校教育費,校外学習費
学習費総額,学校教育費,学校教育費,学級・児童会・生徒会費
学習費総額,学校教育費,学校教育費,その他の学校納付金
学習費総額,学校教育費,学校教育費,PTA会費
学習費総額,学校教育費,学校教育費,後援会費
学習費総額,学校教育費,学校教育費,寄附金
学習費総額,学校教育費,学校教育費,教科書費・教科書以外の図書費
学習費総額,学校教育費,学校教育費,学用品・実験実習材料費
学習費総額,学校教育費,学校教育費,教科外活動費
学習費総額,学校教育費,学校教育費,通学費
学習費総額,学校教育費,学校教育費,制服
学習費総額,学校教育費,学校教育費,通学用品費
学習費総額,学校教育費,学校教育費,その他
学習費総額,学校給食費,学校給食費,学校給食費
学習費総額,学校外活動費,補助学習費,家庭内学習費
学習費総額,学校外活動費,補助学習費,通信教育・家庭教師費
学習費総額,学校外活動費,補助学習費,学習塾費
学習費総額,学校外活動費,補助学習費,その他
学習費総額,学校外活動費,その他の学校外活動費,体験活動・地域活動
学習費総額,学校外活動費,その他の学校外活動費,芸術文化活動
学習費総額,学校外活動費,その他の学校外活動費,スポーツ・レクリエーション活動
学習費総額,学校外活動費,その他の学校外活動費,国際交流体験活動
学習費総額,学校外活動費,その他の学校外活動費,教養・その他"""
            mapping = pd.read_csv(StringIO(csv_data))
        
        print("学習費カテゴリマッピングテーブルを読み込みました")
        return mapping
    except Exception as e:
        print(f"学習費カテゴリマッピングテーブルの読み込みに失敗: {e}")
        raise


def process_tuition_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    「その他」ラベルを曖昧性解消して処理
    
    Parameters
    ----------
    mapping : pd.DataFrame
        元の「その他」ラベルを含む生マッピングテーブル
        
    Returns
    -------
    pd.DataFrame
        曖昧性を解消した「その他」ラベルを含む処理済みマッピングテーブル。
        「その他」ラベルは親カテゴリのコンテキストを含むように更新される
        
    Examples
    --------
    >>> mapping = load_tuition_mapping()
    >>> processed = process_tuition_mapping(mapping)
    >>> # 「その他」が「その他学校教育費」「その他補助学習費」に変更される
    >>> any('その他学校教育費' in val for val in processed['level4'].values)
    True
    """
    try:
        mapping_processed = mapping.copy()
        
        # 親カテゴリのコンテキストを追加して「その他」ラベルを曖昧性解消
        mask1 = (mapping_processed.level3 == "学校教育費") & (mapping_processed.level4 == "その他")
        mask2 = (mapping_processed.level3 == "補助学習費") & (mapping_processed.level4 == "その他")
        
        mapping_processed.loc[mask1, "level4"] = "その他学校教育費"
        mapping_processed.loc[mask2, "level4"] = "その他補助学習費"
        
        print("学習費カテゴリマッピングテーブルを処理しました")
        return mapping_processed
    except Exception as e:
        print(f"学習費カテゴリマッピングテーブルの処理に失敗: {e}")
        raise


def load_tuition_education_data() -> Dict[str, pd.DataFrame]:
    """
    全教育段階の学習費データを読み込み
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        各教育段階のDataFrameを含む辞書。
        キー: 'kindergarten', 'elementary', 'middle', 'high'
        各DataFrameは多層ヘッダーの学習費データを含む
        
    Examples
    --------
    >>> data = load_tuition_education_data()
    >>> list(data.keys())
    ['kindergarten', 'elementary', 'middle', 'high']
    >>> 'elementary' in data
    True
    >>> # 小学校データの列数確認
    >>> data['elementary'].shape[1] > 0
    True

    Note
    --------
    openpyxlのインストールが必要
    """
    urls = get_tuition_urls()
    data = {}
    
    for level in ['kindergarten', 'elementary', 'middle', 'high']:
        try:
            print(f"{level}の学習費データを読み込み中...")
            
            df = pd.read_excel(
                urls[level],
                skiprows=TUITION_SKIPROWS[level],
                header=[0, 1, 2],
                index_col=0,
                na_values=TUITION_NA_VALUES
            )
            
            # 教育段階に基づいた列フィルタリング
            header_col = TUITION_AGE_OR_GRADE[level]
            df = df.loc[:, pd.IndexSlice[:, header_col, :]]
            
            data[level] = df
            print(f"{level}の学習費データを正常に読み込みました")
            
        except Exception as e:
            print(f"{level}の学習費データ読み込みに失敗: {e}")
            raise
    
    return data


def convert_tuition_to_tidy(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    学習費データを整理された形式に変換
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        各教育段階のDataFrame辞書
        
    Returns
    -------
    pd.DataFrame
        整理された形式の結合DataFrame。
        カラム: ['学習費区分', '学校', '学年', '学習費']
        
    Examples
    --------
    >>> # サンプルデータでの例
    >>> sample_data = {
    ...     'kindergarten': pd.DataFrame({
    ...         ('公立幼稚園', '年齢別', '3歳'): [100],
    ...         ('公立幼稚園', '年齢別', '4歳'): [200]
    ...     }, index=['学習費総額'])
    ... }
    >>> tidy_df = convert_tuition_to_tidy(sample_data)
    >>> len(tidy_df.columns) == 4
    True
    >>> '学習費区分' in tidy_df.columns
    True
    """
    tidy_dfs = []
    
    for level, df in data.items():
        try:
            # スタックして整理された形式を作成
            tidy_df = (df.stack([0, 2], future_stack=True)
                      .reset_index()
                      .set_axis(TUITION_TIDY_COLUMNS, axis=1))
            
            tidy_dfs.append(tidy_df)
            print(f"{level}の学習費データを整理された形式に変換しました")
            
        except Exception as e:
            print(f"{level}の学習費データの整理された形式への変換に失敗: {e}")
            raise
    
    return pd.concat(tidy_dfs, ignore_index=True)


def process_tuition_grade_and_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    学年と年齢情報を処理
    
    Parameters
    ----------
    df : pd.DataFrame
        '学年'列に元の学年文字列を含むDataFrame
        
    Returns
    -------
    pd.DataFrame
        処理された学年と年齢列を持つDataFrame:
        - '学年': 標準化された1ベースの学年番号
        - '年齢': 学校種別と学年に基づいて計算された年齢
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     '学年': ['3歳', '第1学年'], 
    ...     '学校': ['幼稚園', '小学校']
    ... })
    >>> result = process_tuition_grade_and_age(df)
    >>> result['年齢'].tolist()
    [3, 6]
    >>> result['学年'].tolist()
    [1, 1]
    """
    df_copy = df.copy()
    
    # 元の学年文字列を保存
    df_copy["orig_grade"] = df_copy["学年"]
    
    # 年齢と学年情報を抽出
    grade_info = df_copy["orig_grade"].apply(extract_tuition_grade_info)
    df_copy["age_extracted"] = [info[0] for info in grade_info]
    df_copy["grade_extracted"] = [info[1] for info in grade_info]
    
    # 標準化された学年を計算（1ベース）
    df_copy["学年"] = np.where(
        df_copy["grade_extracted"].notna(),
        df_copy["grade_extracted"],
        df_copy["age_extracted"] - TUITION_AGE_OFFSETS['kindergarten']
    ).astype(int)
    
    # 年齢を計算
    df_copy["年齢"] = np.where(
        df_copy["age_extracted"].notna(),
        df_copy["age_extracted"],
        calculate_tuition_age_vectorized(df_copy)
    ).astype(int)
    
    # 一時的な列をクリーンアップ
    df_copy = df_copy.drop(columns=["orig_grade", "age_extracted", "grade_extracted"])
    
    print("学年と年齢の処理が完了しました")
    return df_copy


def merge_tuition_with_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    学習費データをカテゴリマッピングとマージ
    
    Parameters
    ----------
    df : pd.DataFrame
        '学習費区分'列を持つ処理済み学習費データ
    mapping : pd.DataFrame
        階層レベルを持つカテゴリマッピングテーブル
        
    Returns
    -------
    pd.DataFrame
        カテゴリ階層列を持つマージされたDataFrame:
        ['学習費区分1', '学習費区分2', '学習費区分3', '学習費区分4']
        マッピングされていないカテゴリの行は削除される
        
    Examples
    --------
    >>> df = pd.DataFrame({'学習費区分': ['学習費総額'], '学習費': [1000]})
    >>> mapping = pd.DataFrame({
    ...     'level1': ['学習費総額'], 'level2': ['学校教育費'], 
    ...     'level3': ['学校教育費'], 'level4': ['学習費総額']
    ... })
    >>> result = merge_tuition_with_mapping(df, mapping)
    >>> '学習費区分1' in result.columns
    True
    """
    df_merged = df.merge(
        mapping[["level1", "level2", "level3", "level4"]],
        left_on="学習費区分",
        right_on="level4",
        how="left"
    )
    
    # すべてのレベル列がnullの行を削除
    df_merged = df_merged.dropna(
        subset=['level1', 'level2', 'level3', 'level4'], 
        how='all'
    )
    
    # 元のカテゴリ列を削除
    df_merged = df_merged.drop(columns=['学習費区分'])
    
    # レベル列の名前を変更
    df_merged = df_merged.rename(columns={
        'level1': '学習費区分1',
        'level2': '学習費区分2',
        'level3': '学習費区分3',
        'level4': '学習費区分4',
    })
    
    print("学習費データをマッピングと正常にマージしました")
    return df_merged


def get_processed_tuition_data() -> pd.DataFrame:
    """
    完全に処理された学習費データを取得
    
    Returns
    -------
    pd.DataFrame
        分析準備完了の学習費データ。
        カラム: ['公立・私立区分', '学校', '学年', '年齢', '学習費区分1', 
                '学習費区分2', '学習費区分3', '学習費区分4', '学習費']
        
    Examples
    --------
    >>> df = get_processed_tuition_data()
    >>> required_cols = ['公立・私立区分', '学校', '学年', '年齢', 
    ...                  '学習費区分1', '学習費区分2', '学習費区分3', 
    ...                  '学習費区分4', '学習費']
    >>> all(col in df.columns for col in required_cols)
    True
    >>> df['公立・私立区分'].nunique() == 2  # 公立・私立
    True
    """
    print("学習費データの処理を開始します...")
    
    # 1. カテゴリマッピングを準備
    raw_mapping = load_tuition_mapping()
    processed_mapping = process_tuition_mapping(raw_mapping)
    
    # 2. 教育データを読み込み
    education_data = load_tuition_education_data()
    
    # 3. インデックスラベルを更新
    education_data_updated = {}
    for level in education_data:
        education_data_updated[level] = update_tuition_index_labels(
            education_data[level]
        )
    
    # 4. 整理された形式に変換
    df_tidy = convert_tuition_to_tidy(education_data_updated)
    
    # 5. 学校情報を分割
    df_split = split_tuition_school_info(df_tidy)
    
    # 6. 学年と年齢を処理
    df_processed = process_tuition_grade_and_age(df_split)
    
    # 7. マッピングとマージ
    df_merged = merge_tuition_with_mapping(df_processed, processed_mapping)
    
    # 8. 最終データセットを準備
    df_final = df_merged[
        ['公立・私立区分', '学校', '学年', '年齢', '学習費区分1', 
         '学習費区分2', '学習費区分3', '学習費区分4', '学習費']
    ].fillna(0).sort_values(['公立・私立区分', '年齢'])
    
    print("学習費データの処理が完了しました")
    return df_final


def create_tuition_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    学習費データのピボットテーブルを作成
    
    Parameters
    ----------
    df : pd.DataFrame
        処理済みの学習費データ
        
    Returns
    -------
    pd.DataFrame
        学習費区分を列とし、教育段階情報を行とするピボットテーブル
        
    Examples
    --------
    >>> df = get_processed_tuition_data()
    >>> pivot_table = create_tuition_pivot_table(df)
    >>> isinstance(pivot_table.columns, pd.MultiIndex)
    True
    >>> len(pivot_table.index.names) == 4  # 公立・私立区分, 学校, 学年, 年齢
    True
    """
    # 学習費区分1～4を行のMultiIndex、指定の列をカラムに、学習費を値にするピボット
    df_table = df.pivot(
        index=['公立・私立区分', '学校', '学年', '年齢'],
        columns=['学習費区分1', '学習費区分2', '学習費区分3', '学習費区分4'],
        values='学習費'
    )
    
    # 列とインデックスをソート
    df_table = df_table.sort_index(level=[0, 3], axis=0).sort_index(axis=1)
    
    print("学習費データのピボットテーブルを作成しました")
    return df_table
