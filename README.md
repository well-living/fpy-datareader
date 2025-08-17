# fpy-datareader

日本のファイナンシャルプランニング（FP）に必要な統計データを簡単に取得・処理するPythonライブラリです。

## 概要

fpy-datareaderは、[jpy-datareader](https://github.com/well-living/jpy-datareader)をベースとして、日本の公的統計データから将来収入予測や教育費分析などのFP業務で必要となるデータを効率的に取得・処理するためのライブラリです。

## 特徴

- 📊 **年齢別収入データの補間・予測**: 複数の統計調査から年齢別収入カーブを作成
- 💰 **生涯収入の計算**: 現在の年齢・収入から将来の生涯収入を推定
- 🎓 **教育費データの分析**: 文部科学省の学習費調査データから詳細な教育費分析
- 🔧 **簡単なAPI**: 複雑な統計データ処理を簡潔な関数呼び出しで実現
- 📈 **成長率・移動平均の自動計算**: 収入の年次成長率と平滑化された傾向の算出

## インストール

```bash
pip install fpy-datareader
```

### 必要な依存関係

- Python 3.12以上
- jpy-datareader>=0.1.0
- pandas>=2.0.0
- scipy>=1.16.0
- openpyxl（教育費データ処理用）

### e-Stat APIキーの取得

このライブラリを使用するには、[e-Stat API](https://www.e-stat.go.jp/api/)でAPIキーを取得する必要があります。

## 使用方法

### 1. 年齢別収入データの分析

#### 賃金構造基本統計調査データ

```python
from fpy_datareader.future_income_estimator import process_wage_structure_survey_data

# APIキーを設定
api_key = "your_estat_api_key"

# 全国の年齢別収入データを取得・処理
income_data = process_wage_structure_survey_data(
    api_key=api_key,
    start_age=20,
    end_age=65,
    include_growth_rate=True,
    include_growth_ma=True
)

print(income_data.head(10))
```

#### 都道府県別の収入データ

```python
# 東京都の年齢別収入データ
tokyo_income = process_wage_structure_survey_data_by_prefecture(
    api_key=api_key,
    area="13000",  # 東京都のコード
    start_age=20,
    end_age=65
)
```

#### 家計調査データ

```python
from fpy_datareader.future_income_estimator import process_family_income_expenditure_survey_data

# 家計調査から世帯収入データを取得
household_income = process_family_income_expenditure_survey_data(
    api_key=api_key,
    yyyy=2024,
    start_age=25,
    end_age=70
)
```

### 2. 生涯収入の計算

```python
from fpy_datareader.future_income_estimator import (
    add_income_multiplier_columns,
    calculate_lifetime_income
)

# 現在30歳、年収500万円の人の生涯収入を計算
lifetime_income = calculate_lifetime_income(
    income_data,
    current_age=30,
    current_income=5000000,
    retirement_age=65,
    rate_column='growth_rate_ma'
)

print(f"推定生涯収入: {lifetime_income:,.0f}円")

# 年齢別の収入倍率を計算
income_with_multiplier = add_income_multiplier_columns(
    income_data,
    start_age=30,
    end_age=65,
    rate_column='growth_rate_ma'
)
```

### 3. 教育費データの分析

```python
from fpy_datareader.child_educational_expenses import get_processed_tuition_data

# 子供の学習費調査データを取得・処理
education_costs = get_processed_tuition_data()

# 公立・私立別の年齢別教育費
print(education_costs.groupby(['公立・私立区分', '年齢'])['学習費'].sum())

# 学校種別の平均費用
print(education_costs.groupby('学校')['学習費'].mean())
```

### 4. 統合的なデータ作成

```python
from fpy_datareader.future_income_estimator import create_age_income_table

# データソースを指定してワンストップで年齢-収入テーブルを作成
income_table = create_age_income_table(
    data_source='wage_structure',
    api_key=api_key,
    start_age=22,
    end_age=65,
    method='cubic_spline',
    include_growth_rate=True
)
```

## 対応データソース

### 収入関連データ

| データソース | 実施機関 | 特徴 | 関数名 |
|------------|----------|-----|--------|
| 賃金構造基本統計調査 | 厚生労働省 | 最も詳細な個人ベース賃金統計 | `process_wage_structure_survey_data` |
| 賃金構造基本統計調査（都道府県別） | 厚生労働省 | 地域別の詳細賃金データ | `process_wage_structure_survey_data_by_prefecture` |
| 家計調査 | 総務省統計局 | 世帯の実収入・支出 | `process_family_income_expenditure_survey_data` |
| 全国家計構造調査 | 総務省統計局 | 5年ごとの詳細な所得・資産調査 | `process_family_income_consumption_wealth_survey_data` |
| 国民生活基礎調査 | 厚生労働省 | 世帯の所得・生活実態 | `process_living_conditions_survey_data` |

### 教育費関連データ

| データソース | 実施機関 | 特徴 | 関数名 |
|------------|----------|-----|--------|
| 子供の学習費調査 | 文部科学省 | 幼稚園から高校までの詳細な学習費 | `get_processed_tuition_data` |

## 出力データ形式

### 年齢別収入データ

```python
age  income     growth_rate  growth_rate_ma
20   3500000    0.000000     0.000000
21   3650000    0.042857     0.028571
22   3800000    0.041096     0.041984
...
```

- `age`: 年齢
- `income`: 年収（円）
- `growth_rate`: 前年からの成長率
- `growth_rate_ma`: 移動平均された成長率

### 教育費データ

```python
公立・私立区分  学校    学年  年齢  学習費区分1   ...  学習費
公立          幼稚園   1    3   学習費総額    ...  223647
公立          幼稚園   1    3   学習費総額    ...  88818
...
```

## 補間・予測機能

### 線形補間 vs 3次スプライン補間

```python
# 線形補間（デフォルト）
linear_data = interpolate_age_income(
    data,
    method='linear',
    start_age=20,
    end_age=65
)

# より滑らかな3次スプライン補間
spline_data = interpolate_age_income(
    data,
    method='cubic_spline',
    start_age=20,
    end_age=65
)
```

### 範囲外補間の制御

```python
# 定数補間（端点の値を維持）
constant_extrapolation = interpolate_age_income(
    data,
    extrapolation='constant'
)

# ターミナル補間（0歳と100歳で0円に設定）
terminal_extrapolation = interpolate_age_income(
    data,
    extrapolation='terminal',
    terminal_start=0,
    terminal_end=100
)
```

## 実用例

### 1. 地域別生涯収入比較

```python
# 東京と大阪の生涯収入を比較
tokyo_data = process_wage_structure_survey_data_by_prefecture(
    api_key, area="13000"  # 東京
)
osaka_data = process_wage_structure_survey_data_by_prefecture(
    api_key, area="27000"  # 大阪
)

tokyo_lifetime = calculate_lifetime_income(
    tokyo_data, current_age=25, current_income=4000000, retirement_age=65
)
osaka_lifetime = calculate_lifetime_income(
    osaka_data, current_age=25, current_income=4000000, retirement_age=65
)

print(f"東京の生涯収入: {tokyo_lifetime:,.0f}円")
print(f"大阪の生涯収入: {osaka_lifetime:,.0f}円")
print(f"差額: {tokyo_lifetime - osaka_lifetime:,.0f}円")
```

### 2. 教育費シミュレーション

```python
# 子供の教育費を年齢別に算出
education_data = get_processed_tuition_data()

# 公立・私立別の総教育費（3歳から18歳まで）
public_total = education_data[
    (education_data['公立・私立区分'] == '公立') &
    (education_data['年齢'].between(3, 18))
]['学習費'].sum()

private_total = education_data[
    (education_data['公立・私立区分'] == '私立') &
    (education_data['年齢'].between(3, 18))
]['学習費'].sum()

print(f"公立コース総教育費: {public_total:,.0f}円")
print(f"私立コース総教育費: {private_total:,.0f}円")
print(f"差額: {private_total - public_total:,.0f}円")
```

## エラーハンドリング

```python
try:
    income_data = process_wage_structure_survey_data(api_key=api_key)
except ValueError as e:
    print(f"パラメータエラー: {e}")
except Exception as e:
    print(f"データ取得エラー: {e}")
```

## 注意事項

1. **APIキーの管理**: e-Stat APIキーは適切に管理し、公開リポジトリにコミットしないでください
2. **API利用制限**: e-Stat APIには利用制限があります。大量データ取得時は適切な間隔を空けてください
3. **データの解釈**: 各調査は対象範囲や調査手法が異なります。複数の調査結果を比較する際は特性を理解してください
4. **プライバシー**: 実在する個人の収入データを扱う際は、プライバシーに十分配慮してください

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 関連プロジェクト

- [jpy-datareader](https://github.com/well-living/jpy-datareader): 日本の統計データ取得の基盤ライブラリ

## クレジット
このサービスは、政府統計総合窓口(e-Stat)のAPI機能を使用していますが、サービスの内容は国によって保証されたものではありません。
https://www.e-stat.go.jp/api/api-info/credit
