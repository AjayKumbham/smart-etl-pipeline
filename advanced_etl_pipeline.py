import pandas as p
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import logging
import yaml
from datetime import datetime
import os

class AdvancedETLPipeline:
    def __init__(self, config=None):
        self.logger = self._setup_logger()
        self.config = config or {}
        self._initialize_transformers()
        
    def _setup_logger(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = f'etl_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, log_file)),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _initialize_transformers(self):
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.simple_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_dataset(self, df):
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Calculate correlation for numeric columns
        numeric_df = df[analysis['numeric_columns']]
        if len(numeric_df.columns) > 1:
            analysis['correlation_matrix'] = numeric_df.corr().to_dict()
            
        # Detect potential outliers in numeric columns
        analysis['outliers'] = {}
        for col in analysis['numeric_columns']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            analysis['outliers'][col] = outliers
            
        return analysis

    def load_data(self, file_path, **kwargs):
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_extension == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.logger.info(f"Successfully loaded data from {file_path}")
            self.logger.info(f"Dataset shape: {df.shape}")
            
            # Analyze dataset
            analysis = self.analyze_dataset(df)
            self.logger.info("Dataset Analysis:")
            self.logger.info(f"Total rows: {analysis['total_rows']}")
            self.logger.info(f"Total columns: {analysis['total_columns']}")
            self.logger.info(f"Memory usage: {analysis['memory_usage']:.2f} MB")
            self.logger.info(f"Duplicate rows: {analysis['duplicate_rows']}")
            
            return df, analysis
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df, strategy='auto'):
        try:
            df_cleaned = df.copy()
            
            if strategy == 'drop':
                df_cleaned = df_cleaned.dropna()
                self.logger.info(f"Dropped {len(df) - len(df_cleaned)} rows with missing values")
                return df_cleaned
            
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            
            # Handle numeric columns
            if len(numeric_columns) > 0:
                if strategy == 'knn':
                    # Ensure data is finite before KNN imputation
                    for col in numeric_columns:
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    df_cleaned[numeric_columns] = self.knn_imputer.fit_transform(df_cleaned[numeric_columns])
                else:  # simple or auto
                    for col in numeric_columns:
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                        median_val = df_cleaned[col].median()
                        df_cleaned[col].fillna(median_val, inplace=True)
            
            # Handle categorical columns
            for col in categorical_columns:
                mode_value = df_cleaned[col].mode()
                if not mode_value.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
            
            self.logger.info("Successfully handled missing values")
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise

    def handle_outliers(self, df, method='isolation_forest', threshold=3):
        try:
            df_cleaned = df.copy()
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_columns) == 0:
                return df_cleaned
            
            # Skip target column if present
            target_col = None
            if 'target' in numeric_columns:
                target_col = df_cleaned['target'].copy()
                numeric_columns = numeric_columns.drop('target')
                df_cleaned = df_cleaned.drop('target', axis=1)
            
            if method == 'isolation_forest':
                outlier_labels = self.isolation_forest.fit_predict(df_cleaned[numeric_columns])
                df_cleaned = df_cleaned[outlier_labels == 1]
                
            elif method == 'zscore':
                for col in numeric_columns:
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                    df_cleaned = df_cleaned[z_scores < threshold]
                    
            elif method == 'iqr':
                for col in numeric_columns:
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_cleaned = df_cleaned[~((df_cleaned[col] < (Q1 - 1.5 * IQR)) | 
                                            (df_cleaned[col] > (Q3 + 1.5 * IQR)))]
            
            # Add target back if it was present
            if target_col is not None:
                df_cleaned['target'] = target_col[df_cleaned.index]
            
            self.logger.info(f"Removed {len(df) - len(df_cleaned)} outliers using {method} method")
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {str(e)}")
            raise

    def encode_categorical_features(self, df, method='auto'):
        try:
            df_encoded = df.copy()
            categorical_columns = df_encoded.select_dtypes(include=['object']).columns
            
            if len(categorical_columns) == 0:
                return df_encoded
                
            for col in categorical_columns:
                unique_values = df_encoded[col].nunique()
                
                # Use label encoding for all categorical columns
                df_encoded[col] = self.label_encoder.fit_transform(df_encoded[col].astype(str))
            
            self.logger.info("Successfully encoded categorical features")
            return df_encoded
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {str(e)}")
            raise

    def scale_features(self, df, method='standard'):
        try:
            df_scaled = df.copy()
            numeric_columns = df_scaled.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_columns) > 0:
                if method == 'standard':
                    df_scaled[numeric_columns] = self.standard_scaler.fit_transform(df_scaled[numeric_columns])
                elif method == 'robust':
                    df_scaled[numeric_columns] = self.robust_scaler.fit_transform(df_scaled[numeric_columns])
            
            self.logger.info(f"Successfully scaled numeric features using {method} scaling")
            return df_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            raise

    def select_features(self, df, target_column, n_features=10, method='mutual_info'):
        try:
            if target_column not in df.columns:
                self.logger.warning(f"Target column '{target_column}' not found in dataframe")
                return df
                
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            if len(X.columns) <= n_features:
                return df
                
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            else:  # f_classif
                selector = SelectKBest(score_func=f_classif, k=n_features)
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            df_selected = pd.DataFrame(X_selected, columns=selected_features)
            df_selected[target_column] = y
            
            self.logger.info(f"Selected {n_features} best features using {method}")
            return df_selected
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise

    def reduce_dimensions(self, df, n_components=None, variance_ratio=0.95):
        try:
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_columns) <= 2:
                return df
                
            if n_components is None:
                pca = PCA(n_components=variance_ratio, svd_solver='full')
            else:
                pca = PCA(n_components=min(n_components, len(numeric_columns)))
            
            transformed_data = pca.fit_transform(df[numeric_columns])
            
            # Create new dataframe with PCA components
            df_pca = pd.DataFrame(
                transformed_data,
                columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])],
                index=df.index
            )
            
            # Add non-numeric columns back if any
            non_numeric = df.select_dtypes(exclude=['int64', 'float64'])
            if not non_numeric.empty:
                df_pca = pd.concat([df_pca, non_numeric], axis=1)
            
            explained_variance = sum(pca.explained_variance_ratio_)
            self.logger.info(f"Reduced dimensions to {transformed_data.shape[1]} components")
            self.logger.info(f"Explained variance ratio: {explained_variance:.2%}")
            
            return df_pca
            
        except Exception as e:
            self.logger.error(f"Error reducing dimensions: {str(e)}")
            raise

    def process_data(self, df, config=None):
        try:
            config = config or {}
            df_processed = df.copy()
            
            # Save and preprocess target column before processing
            target_column = config.get('target_column', 'target')
            target = None
            if target_column in df_processed.columns:
                target = df_processed[target_column].copy()
                # Convert target to numeric if possible
                if target.dtype == 'object':
                    try:
                        target = pd.to_numeric(target, errors='coerce')
                    except:
                        pass
                df_processed = df_processed.drop(target_column, axis=1)
            
            # Remove duplicates
            df_processed = df_processed.drop_duplicates()
            initial_rows = len(df)
            self.logger.info(f"Removed {initial_rows - len(df_processed)} duplicate rows")
            
            # Handle missing values
            missing_strategy = config.get('missing_strategy', 'auto')
            df_processed = self.handle_missing_values(df_processed, strategy=missing_strategy)
            
            # Handle outliers
            outlier_method = config.get('outlier_method', 'isolation_forest')
            df_processed = self.handle_outliers(df_processed, method=outlier_method)
            
            # Keep track of valid indices after outlier removal
            valid_indices = df_processed.index
            
            # Encode categorical features
            encoding_method = config.get('encoding_method', 'auto')
            df_processed = self.encode_categorical_features(df_processed, method=encoding_method)
            
            # Scale features
            scaling_method = config.get('scaling_method', 'standard')
            df_processed = self.scale_features(df_processed, method=scaling_method)
            
            # Optionally reduce dimensions
            if config.get('reduce_dimensions', False):
                variance_ratio = config.get('variance_ratio', 0.95)
                df_processed = self.reduce_dimensions(df_processed, variance_ratio=variance_ratio)
            
            # Add target column back if it exists
            if target is not None:
                # Filter target to match processed data indices
                target = target[valid_indices]
                
                # Handle any NaN in target
                if pd.api.types.is_numeric_dtype(target):
                    # For numeric targets (classification or regression)
                    target_mean = target.mean()
                    target = target.fillna(target_mean)
                    self.logger.info(f"Imputed missing target values with mean")
                else:
                    # For categorical targets
                    mode_val = target.mode().iloc[0] if not target.mode().empty else None
                    target = target.fillna(mode_val)
                    self.logger.info(f"Imputed missing target values with mode")
                
                df_processed[target_column] = target
            
            # Final check for any remaining NaN values
            df_processed = df_processed.dropna()
            self.logger.info(f"Final dataset shape: {df_processed.shape}")
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def save_processed_data(self, df, output_path, save_config=True):
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save processed data
            file_extension = os.path.splitext(output_path)[1].lower()
            if file_extension == '.csv':
                df.to_csv(output_path, index=False)
            elif file_extension in ['.xls', '.xlsx']:
                df.to_excel(output_path, index=False)
            elif file_extension == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)  # Default to CSV
            
            self.logger.info(f"Successfully saved processed data to {output_path}")
            
            # Save configuration if requested
            if save_config and self.config:
                config_path = os.path.splitext(output_path)[0] + '_config.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f)
                self.logger.info(f"Saved configuration to {config_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise
