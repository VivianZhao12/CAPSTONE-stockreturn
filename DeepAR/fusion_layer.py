import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os

class FusionLayer:
    def __init__(self, model_type='rf', lookback_days=7):
        """
        初始化融合层
        
        参数:
        model_type: 融合模型类型 ('rf' 随机森林, 可以扩展其他模型)
        lookback_days: 使用多少天的历史预测作为特征
        """
        self.model_type = model_type
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, daily_predictions_path, quarterly_data_path):
        """
        加载每日预测和季度数据
        """
        try:
            # 加载数据
            print(f"尝试加载文件: {daily_predictions_path}")
            self.daily_df = pd.read_csv(daily_predictions_path)
            print(f"尝试加载文件: {quarterly_data_path}")
            self.quarterly_df = pd.read_csv(quarterly_data_path)
            
            # 确保日期列是datetime格式
            self.daily_df['Date'] = pd.to_datetime(self.daily_df['Date'])
            self.quarterly_df['fiscalDateEnding'] = pd.to_datetime(self.quarterly_df['fiscalDateEnding'])
            
            # 按日期排序
            self.daily_df = self.daily_df.sort_values('Date')
            self.quarterly_df = self.quarterly_df.sort_values('fiscalDateEnding')
            
            print(f"加载了 {len(self.daily_df)} 条每日预测数据")
            print(f"加载了 {len(self.quarterly_df)} 条季度数据")
        except FileNotFoundError as e:
            print(f"错误: 找不到文件 - {e}")
            print(f"当前工作目录: {os.getcwd()}")
            print("请检查文件路径是否正确，或使用绝对路径。")
            raise
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
        
    def merge_quarterly_to_daily(self):
        """
        将季度数据映射到每日数据
        使用前向填充方法，每个日期使用最近的可用季度数据
        """
        # 创建日期范围从daily_df的最早日期到最晚日期
        date_range = pd.date_range(start=self.daily_df['Date'].min(), 
                                   end=self.daily_df['Date'].max())
        date_df = pd.DataFrame({'Date': date_range})
        
        # 将季度数据重命名为方便合并
        quarterly_renamed = self.quarterly_df.rename(columns={
            'fiscalDateEnding': 'Date'
        })
        
        # 将季度数据与日期范围合并
        merged_df = pd.merge_asof(date_df.sort_values('Date'), 
                                  quarterly_renamed.sort_values('Date'),
                                  on='Date', 
                                  direction='backward')
        
        # 将季度数据与每日预测合并
        self.merged_df = pd.merge(self.daily_df, merged_df, on='Date', how='left')
        print(f"合并后的数据有 {len(self.merged_df)} 行，{self.merged_df.shape[1]} 列")
        
    def create_features(self):
        """
        创建用于融合模型的特征
        包括:
        1. 原始DeepAR预测
        2. 滚动平均预测误差
        3. 最近的宏观/微观指标
        4. 时间特征 (季节性, 趋势等)
        5. 预测的历史表现
        """
        df = self.merged_df.copy()
        
        # 创建时间特征
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter

        # 添加波动性特征
        df['recent_volatility'] = df['prediction'].rolling(window=10).std()
        
        # 创建滚动特征
        for window in [3, 7, 14]:
            # 滚动平均预测
            df[f'prediction_rolling_{window}d'] = df['prediction'].rolling(window=window).mean()
            
            # 滚动平均误差
            if 'error' in df.columns:
                df[f'error_rolling_{window}d'] = df['error'].rolling(window=window).mean()
                df[f'abs_error_rolling_{window}d'] = df['abs_error'].rolling(window=window).mean()
            else:
                df[f'error_rolling_{window}d'] = 0
                df[f'abs_error_rolling_{window}d'] = 0
        
        # 使用正确的列名计算财务比率
        # 检查列是否存在
        revenue_col = 'INCOME_STATEMENT_totalRevenue'
        profit_col = 'INCOME_STATEMENT_grossProfit'
        cost_col = 'INCOME_STATEMENT_costOfRevenue'
        
        if revenue_col in df.columns and profit_col in df.columns:
            # 计算季度收入和利润的同比增长率
            df['revenue_yoy_growth'] = df.groupby(['quarter'])[revenue_col].pct_change(4)
            df['profit_yoy_growth'] = df.groupby(['quarter'])[profit_col].pct_change(4)
            
            # 计算利润率
            df['profit_margin'] = df[profit_col] / df[revenue_col]
            
            if cost_col in df.columns:
                # 尝试提取财务数据相对数量级，处理大数字
                df['revenue_to_cost_ratio'] = df[revenue_col] / df[cost_col]
        
        # 添加宏观经济指标（如果存在）
        macro_cols = ['GDP', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL']
        for col in macro_cols:
            if col in df.columns:
                # 创建一些简单的派生特征
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_change'] = df[col].pct_change()
        
        # 添加滞后特征 - 过去几天的预测和误差
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df['prediction'].shift(lag)
            if 'error' in df.columns:
                df[f'error_lag_{lag}'] = df['error'].shift(lag)
            else:
                df[f'error_lag_{lag}'] = 0
        
        # 检查缺失值
        print(f"特征工程前数据有 {len(df)} 行")
        
        # 打印缺失值比例最高的列
        na_ratio = df.isna().mean().sort_values(ascending=False)
        print("\n缺失值比例最高的10个列:")
        print(na_ratio.head(10))
        
        # 先尝试填充滚动特征和滞后特征中的缺失值
        for window in [3, 7, 14]:
            df[f'prediction_rolling_{window}d'] = df[f'prediction_rolling_{window}d'].fillna(df['prediction'])
            if 'error' in df.columns:
                df[f'error_rolling_{window}d'] = df[f'error_rolling_{window}d'].fillna(0)
                df[f'abs_error_rolling_{window}d'] = df[f'abs_error_rolling_{window}d'].fillna(0)
                
        # 填充滞后特征
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df[f'prediction_lag_{lag}'].fillna(df['prediction'])
            if 'error' in df.columns:
                df[f'error_lag_{lag}'] = df[f'error_lag_{lag}'].fillna(0)
        
        # 对财务和宏观数据的缺失值使用向前填充
        finance_cols = [col for col in df.columns if 'INCOME_STATEMENT_' in col or 'BALANCE_SHEET_' in col or 'CASH_FLOW_' in col]
        macro_cols = ['GDP', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL', 'M2SL', 'M1SL', 'PPIACO', 'RTWEXBGS']
        fill_forward_cols = finance_cols + macro_cols
        
        for col in fill_forward_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # 删除任何仍然包含NaN值的行
        self.feature_df = df.dropna()
        print(f"\n特征工程后剩余 {len(self.feature_df)} 行数据")
        
        # 如果没有足够的数据，尝试更激进的缺失值处理
        if len(self.feature_df) < 30:  # 至少需要30行进行训练
            print("数据行数不足，尝试更激进的缺失值处理...")
            # 计算列的缺失率
            missing_ratio = df.isna().mean()
            
            # 删除缺失率超过50%的列
            cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
            print(f"删除以下高缺失率列: {cols_to_drop}")
            df_reduced = df.drop(columns=cols_to_drop)
            
            # 对数值列使用0填充，对非数值列使用众数填充
            numeric_cols = df_reduced.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                df_reduced[col] = df_reduced[col].fillna(0)
                
            categorical_cols = df_reduced.select_dtypes(exclude=['float64', 'int64']).columns
            for col in categorical_cols:
                if col != 'Date':  # 不处理日期列
                    try:
                        mode_value = df_reduced[col].mode()[0]
                        df_reduced[col] = df_reduced[col].fillna(mode_value)
                    except:
                        pass
            
            self.feature_df = df_reduced
            print(f"更激进的处理后剩余 {len(self.feature_df)} 行数据")
            
        # 如果仍然没有足够的数据，使用非常激进的方法
        if len(self.feature_df) < 30:
            print("数据仍然不足，使用最小特征集...")
            min_features_df = df[['Date', 'prediction', 'day_of_week', 'month', 'quarter']].copy()
            
            if 'actual' in df.columns:
                min_features_df['actual'] = df['actual']
                
            # 如果有其他关键特征，添加它们
            for col in ['GDP', 'UNRATE']:
                if col in df.columns:
                    min_features_df[col] = df[col].fillna(df[col].median())
                    
            self.feature_df = min_features_df.dropna()
            print(f"最小特征集后剩余 {len(self.feature_df)} 行数据")
        
    def train_model(self, test_size=0.2):
        """
        训练融合模型
        """
        if len(self.feature_df) == 0:
            raise ValueError("没有足够的数据进行训练，请检查数据处理步骤")
            
        # 确保数据集大小足够进行分割
        if len(self.feature_df) < 5:
            print("警告：数据集太小，无法分割。使用全部数据训练。")
            test_size = 0
            
        # 定义特征和目标列
        exclude_cols = [
            'Date', 'actual', 'error', 'abs_error', 'correct_direction', 
            'Ticker', 'fiscalDateEnding', 'observation_date'
        ]
        
        # 确保'actual'列存在
        if 'actual' not in self.feature_df.columns:
            raise ValueError("数据中缺少'actual'列，无法训练模型")
            
        # 动态确定特征列
        features = [col for col in self.feature_df.columns if col not in exclude_cols]
        
        X = self.feature_df[features]
        y = self.feature_df['actual']
        
        # 处理可能的特征问题
        # 删除所有值都相同的列（这些列没有信息）
        constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        if constant_cols:
            print(f"删除以下常数列: {constant_cols}")
            X = X.drop(columns=constant_cols)
            
        # 检查是否还有特征
        if X.shape[1] == 0:
            raise ValueError("处理后没有可用特征")
            
        # 分割训练和测试集（如果测试集大小大于0）
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X.iloc[:1], y.iloc[:1]  # 创建一个最小的测试集
        
        # 标准化数值特征
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X_train[numeric_features])
        X_train[numeric_features] = self.scaler.transform(X_train[numeric_features])
        X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        # 保存使用的特征列表，以便在预测时使用相同的特征
        self.used_features = X_train.columns.tolist()
        
        # 初始化和训练模型
        
        # 标准化数值特征
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X_train[numeric_features])
        X_train[numeric_features] = self.scaler.transform(X_train[numeric_features])
        X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        # 初始化和训练模型
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # 可以添加其他模型类型
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        
        # 特征重要性
        if self.model_type == 'rf':
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n特征重要性:")
            print(feature_importance.head(10))
        
        print(f"\n训练集 RMSE: {train_rmse:.6f}")
        print(f"测试集 RMSE: {test_rmse:.6f}")
        print(f"训练集 MAE: {train_mae:.6f}")
        print(f"测试集 MAE: {test_mae:.6f}")
        
        # 保存评估结果
        self.evaluation = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
        }
        
        return self.evaluation
    
    def predict(self, new_data=None):
        """
        使用训练好的模型进行预测，并与DeepAR预测融合
        如果提供new_data，则使用它，否则使用训练数据
        """
        # 存储训练时使用的特征
        if not hasattr(self, 'used_features'):
            raise ValueError("模型尚未训练，无法获取使用的特征")
            
        if new_data is not None:
            # 确保所有训练特征在预测数据中可用
            for feature in self.used_features:
                if feature not in new_data.columns:
                    print(f"警告: 缺少特征 '{feature}'，使用0填充")
                    new_data[feature] = 0
                    
            # 只使用训练时使用的特征
            X = new_data[self.used_features]
            # 获取原始DeepAR预测
            deepar_predictions = new_data['prediction'].values
        else:
            # 使用训练数据
            X = self.feature_df[self.used_features]
            # 获取原始DeepAR预测
            deepar_predictions = self.feature_df['prediction'].values
        
        # 标准化数值特征
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # 获取融合模型的基础预测
        base_predictions = self.model.predict(X)
        
        # 检测可能的波动性
        if 'recent_volatility' in X.columns:
            volatility = X['recent_volatility'].values
            # 动态调整权重 - 高波动期间给DeepAR更多权重
            alpha = np.ones_like(base_predictions) * 0.6  # 默认权重
            high_vol_mask = volatility > np.median(volatility)
            alpha[high_vol_mask] = 0.4  # 高波动期间降低融合模型权重
        else:
            # 使用固定权重
            alpha = 0.6
        
        # 融合预测，保留更多DeepAR的波动性
        final_predictions = alpha * base_predictions + (1 - alpha) * deepar_predictions
        
        return final_predictions
    
    def predict_future(self, future_dates_df, future_quarterly_data=None):
        """
        预测未来时间段
        
        参数:
        future_dates_df: 包含未来日期和DeepAR预测的DataFrame
        future_quarterly_data: 可选的未来季度数据
        
        返回:
        带有融合预测的DataFrame
        """
        print("预测未来时间段...")
        
        # 合并未来的季度数据（如果有）
        if future_quarterly_data is not None:
            # 确保日期格式正确
            future_quarterly_data['fiscalDateEnding'] = pd.to_datetime(future_quarterly_data['fiscalDateEnding'])
            # 添加到现有季度数据
            combined_quarterly = pd.concat([self.quarterly_df, future_quarterly_data]).drop_duplicates()
            # 更新季度数据
            self.quarterly_df = combined_quarterly
        
        # 为未来日期准备数据
        future_df = future_dates_df.copy()
        future_df['Date'] = pd.to_datetime(future_df['Date'])
        
        # 获取最新季度数据并前向填充
        latest_quarterly = self.quarterly_df.sort_values('fiscalDateEnding').iloc[-1:].copy()
        
        # 预测下一个季度的财务数据
        next_quarter_pred = self.predict_next_quarters(num_quarters=4)
        
        # 合并实际的季度数据和预测的季度数据
        all_quarterly = pd.concat([self.quarterly_df, next_quarter_pred])
        
        # 将季度数据映射到未来日期
        quarterly_renamed = all_quarterly.rename(columns={'fiscalDateEnding': 'Date'})
        future_with_quarterly = pd.merge_asof(
            future_df.sort_values('Date'), 
            quarterly_renamed.sort_values('Date'),
            on='Date', 
            direction='backward'
        )
        
        # 创建特征
        future_feature_df = self.prepare_future_features(future_with_quarterly)
        
        # 确保所有需要的特征都存在
        if not hasattr(self, 'used_features'):
            print("警告: 模型未保存使用的特征列表，尝试使用所有可用特征")
            # 使用所有可能的特征
            exclude_cols = [
                'Date', 'actual', 'error', 'abs_error', 'correct_direction', 
                'Ticker', 'fiscalDateEnding', 'observation_date'
            ]
            features = [col for col in future_feature_df.columns if col not in exclude_cols]
        else:
            features = self.used_features
            
        # 确保所有需要的特征都存在
        for feature in features:
            if feature not in future_feature_df.columns:
                print(f"添加缺失特征: {feature}")
                future_feature_df[feature] = 0
        
        # 使用训练好的模型预测
        X = future_feature_df[features]
        
        # 标准化特征
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # 预测
        future_predictions = self.model.predict(X)
        
        # 添加预测结果
        result_df = future_feature_df.copy()
        result_df['fusion_prediction'] = future_predictions
        
        print(f"预测完成，共 {len(result_df)} 条预测")
        
        return result_df
    
    def prepare_future_features(self, future_df):
        """
        为未来数据准备特征
        """
        df = future_df.copy()
        
        # 创建时间特征
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # 创建滚动特征
        for window in [3, 7, 14]:
            # 滚动平均预测
            df[f'prediction_rolling_{window}d'] = df['prediction'].rolling(window=window).mean()
            
            # 如果有历史数据，可以计算历史误差的滚动平均
            if 'error' in df.columns:
                df[f'error_rolling_{window}d'] = df['error'].rolling(window=window).mean()
                df[f'abs_error_rolling_{window}d'] = df['abs_error'].rolling(window=window).mean()
            else:
                # 否则使用0填充
                df[f'error_rolling_{window}d'] = 0
                df[f'abs_error_rolling_{window}d'] = 0
        
        # 使用正确的列名计算财务比率
        revenue_col = 'INCOME_STATEMENT_totalRevenue'
        profit_col = 'INCOME_STATEMENT_grossProfit'
        cost_col = 'INCOME_STATEMENT_costOfRevenue'
        
        if revenue_col in df.columns and profit_col in df.columns:
            try:
                # 计算季度收入和利润的同比增长率
                df['revenue_yoy_growth'] = df.groupby(['quarter'])[revenue_col].pct_change(4, fill_method=None)
                df['profit_yoy_growth'] = df.groupby(['quarter'])[profit_col].pct_change(4, fill_method=None)
                
                # 移除异常值
                df['revenue_yoy_growth'] = df['revenue_yoy_growth'].apply(
                    lambda x: x if (pd.notna(x) and abs(x) < 0.5) else 0
                )
                df['profit_yoy_growth'] = df['profit_yoy_growth'].apply(
                    lambda x: x if (pd.notna(x) and abs(x) < 0.5) else 0
                )
                
                # 计算利润率
                df['profit_margin'] = df.apply(
                    lambda row: row[profit_col] / row[revenue_col] if row[revenue_col] != 0 else 0, 
                    axis=1
                )
                
                if cost_col in df.columns:
                    # 尝试提取财务数据相对数量级
                    df['revenue_to_cost_ratio'] = df.apply(
                        lambda row: row[revenue_col] / row[cost_col] if row[cost_col] != 0 else 1, 
                        axis=1
                    )
            except Exception as e:
                print(f"计算财务比率时出错：{e}")
                # 如果计算失败，使用默认值
                df['revenue_yoy_growth'] = 0
                df['profit_yoy_growth'] = 0
                df['profit_margin'] = 0
                df['revenue_to_cost_ratio'] = 1
        
        # 添加宏观经济指标（如果存在）
        macro_cols = ['GDP', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL']
        for col in macro_cols:
            if col in df.columns:
                try:
                    # 创建一些简单的派生特征
                    df[f'{col}_lag1'] = df[col].shift(1)
                    df[f'{col}_change'] = df[col].pct_change(fill_method=None)
                    
                    # 将NaN和无穷大替换为0
                    df[f'{col}_change'] = df[f'{col}_change'].apply(
                        lambda x: x if (pd.notna(x) and abs(x) < 0.5) else 0
                    )
                except:
                    df[f'{col}_lag1'] = df[col]
                    df[f'{col}_change'] = 0
        
        # 添加滞后特征 - 使用可用的最近历史数据
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df['prediction'].shift(lag)
            
            if 'error' in df.columns:
                df[f'error_lag_{lag}'] = df['error'].shift(lag)
            else:
                df[f'error_lag_{lag}'] = 0
        
        # 对于未来预测的第一个日期，可能需要从历史数据获取滞后特征
        # 这里简化处理，填充缺失值
        # 对滚动和滞后特征进行填充
        for window in [3, 7, 14]:
            df[f'prediction_rolling_{window}d'] = df[f'prediction_rolling_{window}d'].fillna(df['prediction'])
            df[f'error_rolling_{window}d'] = df[f'error_rolling_{window}d'].fillna(0)
            df[f'abs_error_rolling_{window}d'] = df[f'abs_error_rolling_{window}d'].fillna(0)
            
        for lag in range(1, self.lookback_days+1):
            df[f'prediction_lag_{lag}'] = df[f'prediction_lag_{lag}'].fillna(df['prediction'])
            df[f'error_lag_{lag}'] = df[f'error_lag_{lag}'].fillna(0)
            
        # 对其他财务和宏观数据使用向前填充
        for col in df.columns:
            if col not in ['Date', 'prediction'] and df[col].isnull().any():
                df[col] = df[col].ffill().bfill().fillna(0)
                
        # 最终检查：替换任何NaN或无穷大的值
        for col in df.columns:
            if col != 'Date':  # 不处理日期列
                try:
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
                    df[col] = df[col].fillna(0)
                except:
                    pass
                
        return df
    
    def predict_next_quarters(self, num_quarters=4):
        """
        预测未来几个季度的财务数据
        简单方法：使用时序预测或平均增长率
        """
        # 获取最近的季度数据
        recent_quarters = self.quarterly_df.sort_values('fiscalDateEnding').tail(8)
        
        # 计算平均季度增长率
        growth_rates = {}
        for col in recent_quarters.columns:
            if col in ['Ticker', 'fiscalDateEnding', 'observation_date'] or 'reportedCurrency' in col:
                continue
                
            # 只对数值列计算增长率
            if recent_quarters[col].dtype in [np.float64, np.int64]:
                try:
                    # 计算季度环比增长率
                    pct_changes = recent_quarters[col].pct_change().dropna()
                    # 取平均增长率
                    if not pct_changes.empty:
                        growth_rates[col] = pct_changes.mean()
                except:
                    # 如果计算失败，设置为0
                    growth_rates[col] = 0
        
        # 获取最新的季度数据
        last_quarter = recent_quarters.iloc[-1].copy()
        next_quarters = []
        
        # 预测未来几个季度
        for i in range(1, num_quarters+1):
            # 复制上一季度数据
            next_quarter = last_quarter.copy()
            
            # 计算下一季度的日期
            next_date = pd.to_datetime(last_quarter['fiscalDateEnding']) + pd.DateOffset(months=3*i)
            next_quarter['fiscalDateEnding'] = next_date
            
            # 更新财务指标，应用增长率
            for col, rate in growth_rates.items():
                if pd.notna(last_quarter[col]) and last_quarter[col] != 0:
                    next_quarter[col] = last_quarter[col] * (1 + rate)
            
            # 添加到预测季度列表
            next_quarters.append(next_quarter)
        
        # 转换为DataFrame
        next_quarters_df = pd.DataFrame(next_quarters)
        return next_quarters_df
    
    def run_pipeline(self, daily_predictions_path, quarterly_data_path, future_days=None, test_size=0.2):
        """
        运行完整的融合模型流程
        
        参数:
        daily_predictions_path: 每日预测数据路径
        quarterly_data_path: 季度数据路径
        future_days: 如果提供，将预测未来的这些天数
        test_size: 训练测试分割比例
        """
        print("1. 加载数据...")
        self.load_data(daily_predictions_path, quarterly_data_path)
        
        print("\n2. 合并季度数据到每日数据...")
        self.merge_quarterly_to_daily()
        
        # 检查合并后的数据列
        print("\n数据列预览:")
        for col in sorted(self.merged_df.columns):
            if 'Date' in col or 'prediction' in col or 'actual' in col or 'error' in col:
                print(f"- {col}")
                
        # 检查是否有必要的列
        required_cols = ['Date', 'prediction']
        for col in required_cols:
            if col not in self.merged_df.columns:
                raise ValueError(f"缺少必要的列: {col}")
                
        print("\n3. 创建融合特征...")
        self.create_features()
        
        # 如果特征创建后行数为0，尝试基本特征
        if len(self.feature_df) == 0:
            print("警告: 特征工程后没有数据，尝试使用基本特征...")
            df = self.merged_df.copy()
            
            # 只保留基本特征
            basic_cols = ['Date', 'prediction']
            
            if 'actual' in df.columns:
                basic_cols.append('actual')
                
            # 添加时间特征
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter
            
            # 使用基本特征作为特征集
            self.feature_df = df[basic_cols + ['day_of_week', 'month', 'quarter']]
            print(f"使用基本特征后的数据行数: {len(self.feature_df)}")
        
        print("\n4. 训练融合模型...")
        try:
            evaluation = self.train_model(test_size=test_size)
        except Exception as e:
            print(f"训练模型失败: {e}")
            # 使用简单的线性回归作为备选
            print("尝试使用简单线性回归作为备选模型...")
            from sklearn.linear_model import LinearRegression
            
            # 使用最基本的特征
            basic_df = self.merged_df[['Date', 'prediction']].copy()
            basic_df['day_of_week'] = basic_df['Date'].dt.dayofweek
            basic_df['month'] = basic_df['Date'].dt.month
            
            if 'actual' in self.merged_df.columns:
                basic_df['actual'] = self.merged_df['actual']
                
            self.feature_df = basic_df.dropna()
            
            # 使用prediction作为唯一特征
            X = self.feature_df[['prediction', 'day_of_week', 'month']]
            y = self.feature_df['actual']
            
            # 训练简单模型
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.model_type = 'linear'
            
            # 简单评估
            preds = self.model.predict(X)
            mse = mean_squared_error(y, preds)
            print(f"简单线性模型MSE: {mse:.6f}")
        
        print("\n5. 生成历史数据的最终预测...")
        final_predictions = self.predict()
        
        # 将预测结果添加到原始数据中
        result_df = self.feature_df.copy()
        result_df['fusion_prediction'] = final_predictions
        
        if 'actual' in result_df.columns:
            result_df['fusion_error'] = result_df['fusion_prediction'] - result_df['actual']
            result_df['fusion_abs_error'] = np.abs(result_df['fusion_error'])
            
            if 'prediction' in result_df.columns:
                result_df['deepar_error'] = result_df['prediction'] - result_df['actual']
                result_df['deepar_abs_error'] = np.abs(result_df['deepar_error'])
                
                # 计算融合模型相对于DeepAR的改进
                avg_deepar_error = result_df['deepar_abs_error'].mean()
                avg_fusion_error = result_df['fusion_abs_error'].mean()
                improvement = (avg_deepar_error - avg_fusion_error) / avg_deepar_error * 100
                
                print(f"\n融合模型平均绝对误差: {avg_fusion_error:.6f}")
                print(f"DeepAR模型平均绝对误差: {avg_deepar_error:.6f}")
                print(f"相对改进: {improvement:.2f}%")
        
        # 只保留Date和fusion_prediction列
        output_df = result_df[['Date', 'fusion_prediction', 'actual']].copy()
        
        # 如果需要预测未来
        if future_days is not None:
            print("\n6. 生成未来预测...")
            # 创建未来日期
            last_date = self.daily_df['Date'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
            
            # 创建未来数据框架
            # 这里假设DeepAR模型已经为这些日期生成了预测
            # 如果没有，则使用最后一个预测值作为占位符
            last_prediction = self.daily_df['prediction'].iloc[-1]
            future_df = pd.DataFrame({
                'Date': future_dates,
                'prediction': [last_prediction] * future_days  # 占位符预测值
            })
            
            # 预测未来
            future_predictions = self.predict_future(future_df)
            future_output = future_predictions[['Date', 'fusion_prediction', 'actual']].copy()
            
            # 合并历史和未来预测
            all_results = pd.concat([output_df, future_output])
            return all_results
        
        return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fusion Layer for Time Series Prediction')
    parser.add_argument('--daily', type=str, required=True, help='Path to daily predictions CSV')
    parser.add_argument('--quarterly', type=str, required=True, help='Path to quarterly data CSV')
    parser.add_argument('--future', type=int, default=0, help='Number of future days to predict')
    parser.add_argument('--output', type=str, default='fusion_predictions.csv', help='Output file path')
    parser.add_argument('--lookback', type=int, default=7, help='Number of lookback days for features')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test data')
    
    args = parser.parse_args()
    
    # 打印参数
    print("运行参数:")
    print(f"每日预测数据: {args.daily}")
    print(f"季度数据: {args.quarterly}")
    print(f"预测未来天数: {args.future}")
    print(f"输出文件: {args.output}")
    print(f"特征历史天数: {args.lookback}")
    print(f"测试集比例: {args.test_size}")
    
    # 创建融合层实例
    fusion = FusionLayer(lookback_days=args.lookback)
    
    # 运行完整流程
    results = fusion.run_pipeline(
        daily_predictions_path=args.daily,
        quarterly_data_path=args.quarterly,
        future_days=args.future if args.future > 0 else None,
        test_size=args.test_size
    )
    
    # 从daily路径中提取目录部分
    output_dir = os.path.dirname(args.daily)
    
    # 将输出文件放在与daily相同的目录中
    output_file = os.path.join(output_dir, args.output)
    
    results.to_csv(output_file, index=False)
    print(f"\n结果已保存到 {output_file}")