import pytest
import pandas as pd
import numpy as np
from fpy_datareader.future_income_estimator import (
    extract_numbers,
    calculate_age_midpoint,
    add_age_midpoint_column,
    interpolate_age_income
)



class TestExtractNumbers:
    """extract_numbers関数のテストクラス"""
    
    def test_extract_single_number(self):
        """単一の数値を抽出"""
        assert extract_numbers("25") == [25]
        assert extract_numbers("100") == [100]
    
    def test_extract_multiple_numbers(self):
        """複数の数値を抽出"""
        assert extract_numbers("25～29") == [25, 29]
        assert extract_numbers("30歳～34歳") == [30, 34]
        assert extract_numbers("5歳階級") == [5]
    
    def test_extract_with_full_width_numbers(self):
        """全角数字を含む文字列から抽出"""
        assert extract_numbers("２５～２９") == [25, 29]
        assert extract_numbers("３０歳未満") == [30]
    
    def test_extract_no_numbers(self):
        """数値が含まれない文字列"""
        assert extract_numbers("平均") == []
        assert extract_numbers("総数") == []
    
    def test_extract_with_mixed_characters(self):
        """数値と文字が混在する文字列"""
        assert extract_numbers("30歳以上40歳未満") == [30, 40]
        assert extract_numbers("年収500万円") == [500]
    
    def test_extract_empty_string(self):
        """空文字列"""
        assert extract_numbers("") == []


class TestCalculateAgeMidpoint:
    """calculate_age_midpoint関数のテストクラス"""
    
    def test_age_range_discrete(self):
        """年齢範囲の離散的解釈"""
        assert calculate_age_midpoint("30～34", method='discrete') == 32.0
        assert calculate_age_midpoint("25～29歳", method='discrete') == 27.0
    
    def test_age_range_continuous(self):
        """年齢範囲の連続的解釈"""
        assert calculate_age_midpoint("30～34", method='continuous') == 32.5
        assert calculate_age_midpoint("25～29歳", method='continuous') == 27.5
    
    def test_age_under_discrete(self):
        """年齢未満の離散的解釈"""
        assert calculate_age_midpoint("30歳未満", method='discrete') == 27.0  # (25+29)/2
        assert calculate_age_midpoint("～19歳", method='discrete') == 17.0    # (15+19)/2
    
    def test_age_under_continuous(self):
        """年齢未満の連続的解釈"""
        assert calculate_age_midpoint("30歳未満", method='continuous') == 27.5  # (25+30)/2
        assert calculate_age_midpoint("～19歳", method='continuous') == 17.5    # (15+20)/2
    
    def test_age_over_discrete(self):
        """年齢以上の離散的解釈"""
        assert calculate_age_midpoint("85歳以上", method='discrete') == 87.0   # (85+89)/2
        assert calculate_age_midpoint("70歳～", method='discrete') == 72.0     # (70+74)/2
    
    def test_age_over_continuous(self):
        """年齢以上の連続的解釈"""
        assert calculate_age_midpoint("85歳以上", method='continuous') == 87.5  # (85+90)/2
        assert calculate_age_midpoint("70歳～", method='continuous') == 72.5    # (70+75)/2
    
    def test_full_width_numbers(self):
        """全角数字を含む年齢階級"""
        assert calculate_age_midpoint("３０～３４", method='discrete') == 32.0
        assert calculate_age_midpoint("２５歳未満", method='discrete') == 22.0
    
    def test_custom_age_class_width(self):
        """カスタム階級幅のテスト"""
        assert calculate_age_midpoint("30歳未満", method='discrete', age_class_width=10) == 24.5  # (20+29)/2
        assert calculate_age_midpoint("85歳以上", method='discrete', age_class_width=10) == 89.5  # (85+94)/2
        
    def test_age_class_with_explicit_width(self):
        """明示的な階級幅を含む年齢階級"""
        assert calculate_age_midpoint("30歳未満（5歳階級）", method='discrete') == 27.0
        assert calculate_age_midpoint("85歳以上（10歳階級）", method='discrete') == 89.5
    
    def test_invalid_patterns(self):
        """無効なパターン"""
        assert calculate_age_midpoint("平均", method='discrete') is None
        assert calculate_age_midpoint("総数", method='discrete') is None
        assert calculate_age_midpoint("", method='discrete') is None
    
    def test_age_below_patterns(self):
        """年齢以下のパターン"""
        assert calculate_age_midpoint("29歳以下", method='discrete') == 27.0    # (25+29)/2
        assert calculate_age_midpoint("19歳以下", method='continuous') == 17.5  # (15+20)/2


class TestAddAgeMidpointColumn:
    """add_age_midpoint_column関数のテストクラス"""
    
    def setup_method(self):
        """テスト用データの準備"""
        self.test_df = pd.DataFrame({
            'age_class': ['25～29歳', '30～34歳', '35～39歳'],
            'income': [400000, 500000, 600000],
            'other_col': ['A', 'B', 'C']
        })
    
    def test_basic_functionality(self):
        """基本的な機能のテスト"""
        result = add_age_midpoint_column(self.test_df, 'age_class', 'income')
        
        assert len(result) == 3
        assert list(result.columns) == ['age', 'income']
        assert result['age'].tolist() == [27.0, 32.0, 37.0]  # discrete method default
        assert result['income'].tolist() == [400000, 500000, 600000]
    
    def test_discrete_method(self):
        """離散的解釈のテスト"""
        result = add_age_midpoint_column(self.test_df, 'age_class', 'income', method='discrete')
        
        assert result['age'].tolist() == [27.0, 32.0, 37.0]
        assert result['income'].tolist() == [400000, 500000, 600000]
    
    def test_continuous_method(self):
        """連続的解釈のテスト"""
        result = add_age_midpoint_column(self.test_df, 'age_class', 'income', method='continuous')
        
        assert result['age'].tolist() == [27.5, 32.5, 37.5]
        assert result['income'].tolist() == [400000, 500000, 600000]
    
    def test_custom_age_class_width(self):
        """カスタム階級幅のテスト"""
        df_custom = pd.DataFrame({
            'age_class': ['30歳未満', '85歳以上'],
            'income': [300000, 200000]
        })
        
        result = add_age_midpoint_column(df_custom, 'age_class', 'income', 
                                       method='discrete', age_class_width=10)
        
        assert result['age'].tolist() == [24.5, 89.5]  # (20+29)/2, (85+94)/2
        assert result['income'].tolist() == [300000, 200000]
    
    def test_with_none_values(self):
        """Noneが含まれる場合のテスト"""
        df_with_none = pd.DataFrame({
            'age_class': ['25～29歳', '平均', '35～39歳'],  # '平均'はNoneになる
            'income': [400000, 500000, 600000]
        })
        
        result = add_age_midpoint_column(df_with_none, 'age_class', 'income')
        
        assert len(result) == 3
        assert result['age'].iloc[0] == 27.0  # discrete method default
        assert pd.isna(result['age'].iloc[1])  # '平均'はNone
        assert result['age'].iloc[2] == 37.0
    
    def test_empty_dataframe(self):
        """空のDataFrameのテスト"""
        empty_df = pd.DataFrame({'age_class': [], 'income': []})
        result = add_age_midpoint_column(empty_df, 'age_class', 'income')
        
        assert len(result) == 0
        assert list(result.columns) == ['age', 'income']


class TestInterpolateAgeIncome:
    """interpolate_age_income関数のテストクラス"""
    
    def setup_method(self):
        """テスト用データの準備"""
        self.sample_data = [
            [25, 300000],
            [30, 400000],
            [35, 500000],
            [40, 550000],
            [45, 600000]
        ]
    
    def test_basic_linear_interpolation(self):
        """基本的な線形補間のテスト"""
        result = interpolate_age_income(self.sample_data, start_age=25, end_age=45, 
                                      method='linear', extrapolation='constant')
        
        assert len(result) == 21  # 25-45歳の21年分
        assert result['age'].tolist() == list(range(25, 46))
        assert result['income'].iloc[0] == 300000  # 25歳
        assert result['income'].iloc[5] == 400000  # 30歳
        assert result['income'].iloc[-1] == 600000  # 45歳
    
    def test_cubic_spline_interpolation(self):
        """3次スプライン補間のテスト"""
        result = interpolate_age_income(self.sample_data, start_age=25, end_age=45,
                                      method='cubic_spline', extrapolation='constant')
        
        assert len(result) == 21
        assert result['age'].tolist() == list(range(25, 46))
        # スプライン補間では厳密な値チェックは困難なので、基本的な構造のみ確認
        assert all(result['income'] >= 0)  # 負の値にならないことを確認
    
    def test_extrapolation_constant(self):
        """定数外挿のテスト"""
        result = interpolate_age_income(self.sample_data, start_age=20, end_age=50,
                                      method='linear', extrapolation='constant')
        
        # 範囲外の値は最端値と同じになる
        assert result.loc[result['age'] == 20, 'income'].iloc[0] == 300000  # 最小値
        assert result.loc[result['age'] == 50, 'income'].iloc[0] == 600000  # 最大値
    
    def test_extrapolation_terminal(self):
        """ターミナル外挿のテスト"""
        result = interpolate_age_income(self.sample_data, start_age=0, end_age=100,
                                      method='linear', extrapolation='terminal',
                                      terminal_start=0, terminal_end=100)
        
        assert len(result) == 101  # 0-100歳の101年分
        assert result.loc[result['age'] == 0, 'income'].iloc[0] == 0
        assert result.loc[result['age'] == 100, 'income'].iloc[0] == 0
    
    def test_growth_rate_calculation(self):
        """成長率計算のテスト"""
        result = interpolate_age_income(self.sample_data, start_age=25, end_age=30,
                                      method='linear', include_growth_rate=True)
        
        assert 'growth_rate' in result.columns
        assert len(result) == 6  # 25-30歳の6年分
        
        # 成長率の基本的な計算確認（年収が増加している場合は正の値）
        growth_rates = result['growth_rate'].dropna()
        assert all(growth_rates >= 0)  # このサンプルデータでは収入が単調増加
    
    def test_growth_rate_moving_average(self):
        """成長率移動平均のテスト"""
        result = interpolate_age_income(self.sample_data, start_age=25, end_age=35,
                                      method='linear', include_growth_rate=True,
                                      include_growth_ma=True, growth_ma_periods=3)
        
        assert 'growth_rate' in result.columns
        assert 'growth_rate_ma' in result.columns
        assert len(result) == 11  # 25-35歳の11年分
    
    def test_dataframe_input(self):
        """DataFrame入力のテスト"""
        df_input = pd.DataFrame(self.sample_data, columns=['age', 'income'])
        result = interpolate_age_income(df_input, start_age=25, end_age=45)
        
        assert len(result) == 21
        assert result['age'].tolist() == list(range(25, 46))
    
    def test_numpy_array_input(self):
        """NumPy配列入力のテスト"""
        np_input = np.array(self.sample_data)
        result = interpolate_age_income(np_input, start_age=25, end_age=45)
        
        assert len(result) == 21
        assert result['age'].tolist() == list(range(25, 46))
    
    def test_negative_income_clipping(self):
        """負の収入値のクリッピングテスト"""
        # 負の値を含むデータ
        negative_data = [
            [25, 100000],
            [30, -50000],  # 負の値
            [35, 200000]
        ]
        
        result = interpolate_age_income(negative_data, start_age=25, end_age=35,
                                      method='linear')
        
        # 全ての収入値が0以上であることを確認
        assert all(result['income'] >= 0)
    
    def test_parameter_validation(self):
        """パラメータ検証のテスト"""
        # start_age < 0
        with pytest.raises(ValueError, match="start_ageは0以上である必要があります"):
            interpolate_age_income(self.sample_data, start_age=-1)
        
        # end_age < start_age
        with pytest.raises(ValueError, match="end_ageはstart_age以上である必要があります"):
            interpolate_age_income(self.sample_data, start_age=30, end_age=25)
        
        # terminal_start < 0
        with pytest.raises(ValueError, match="terminal_startは0以上である必要があります"):
            interpolate_age_income(self.sample_data, terminal_start=-1)
        
        # terminal_end <= terminal_start
        with pytest.raises(ValueError, match="terminal_endはterminal_startより大きい必要があります"):
            interpolate_age_income(self.sample_data, terminal_start=50, terminal_end=50)
        
        # growth_ma_periods < 1
        with pytest.raises(ValueError, match="growth_ma_periodsは1以上である必要があります"):
            interpolate_age_income(self.sample_data, include_growth_rate=True,
                                 include_growth_ma=True, growth_ma_periods=0)
        
        # 無効なmethod
        with pytest.raises(ValueError, match="methodは'linear'または'cubic_spline'を指定してください"):
            interpolate_age_income(self.sample_data, method='invalid')
        
        # 無効なデータ形式
        with pytest.raises(ValueError, match="データ形式が不正です"):
            interpolate_age_income("invalid_data")
    
    def test_age_data_validation(self):
        """年齢データ検証のテスト"""
        # 負の年齢を含むデータ
        invalid_age_data = [
            [-5, 100000],
            [25, 200000]
        ]
        
        with pytest.raises(ValueError, match="年齢データは0以上である必要があります"):
            interpolate_age_income(invalid_age_data)
    
    def test_single_data_point(self):
        """単一データポイントのテスト"""
        single_data = [[30, 400000]]
        result = interpolate_age_income(single_data, start_age=25, end_age=35,
                                      extrapolation='constant')
        
        # 定数補間により全て同じ値になる
        assert all(result['income'] == 400000)
    
    def test_unsorted_age_data(self):
        """ソートされていない年齢データのテスト"""
        unsorted_data = [
            [35, 500000],
            [25, 300000],
            [45, 600000],
            [30, 400000]
        ]
        
        result = interpolate_age_income(unsorted_data, start_age=25, end_age=45)
        
        # データが自動的にソートされることを確認
        assert len(result) == 21
        assert result['age'].tolist() == list(range(25, 46))
        assert result['income'].iloc[0] == 300000  # 25歳
        assert result['income'].iloc[10] == 500000  # 35歳


if __name__ == "__main__":
    pytest.main([__file__])