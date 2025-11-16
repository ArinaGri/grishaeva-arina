"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class PropensityCalculator:
    """Калькулятор пропенсити scores для корректной IPS оценки"""
    def __init__(self):
        self.propensity_model = None
        self.action_mapping = {
            'Womens E-Mail': 0,
            'Mens E-Mail': 1, 
            'No E-Mail': 2
        }
    
    def fit(self, X, actions):
        """Обучаем мультиклассовую модель для оценки пропенсити каждого действия"""
        self.propensity_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        self.propensity_model.fit(X, actions)
        print("Propensity model обучен")
    
    def predict_propensity(self, X, actions=None):
        """Предсказание пропенсити scores"""
        if actions is None:
            return self.propensity_model.predict_proba(X)
        else:
            propensity_matrix = self.propensity_model.predict_proba(X)
            propensities = []
            
            for i, action in enumerate(actions):
                propensities.append(propensity_matrix[i, action])
            
            return np.array(propensities)

class AdvancedDataPreprocessor:
    """Продвинутая предобработка с созданием осмысленных признаков"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def _create_contextual_features(self, df):
        """Создание контекстуальных признаков на основе характеристик клиентов"""
        df_processed = df.copy()
        
        # 1. Взаимодействия признаков (контекстуальное обучение)
        df_processed['history_recency_ratio'] = df_processed['history'] / (df_processed['recency'] + 1)
        df_processed['customer_value_score'] = df_processed['history'] * (1 / (df_processed['recency'] + 0.1))
        
        # 2. Сегментация пользователей
        df_processed['value_segment'] = pd.qcut(df_processed['history'], q=4, 
                                              labels=['low_value', 'medium_value', 'high_value', 'vip'])
        df_processed['activity_segment'] = pd.cut(df_processed['recency'], 
                                                bins=[0, 3, 6, 12], 
                                                labels=['active', 'medium_active', 'inactive'])
        
        # 3. Признаки предпочтений
        df_processed['gender_preference'] = 'none'
        df_processed.loc[df_processed['mens'] == 1, 'gender_preference'] = 'mens'
        df_processed.loc[df_processed['womens'] == 1, 'gender_preference'] = 'womens' 
        df_processed.loc[(df_processed['mens'] == 1) & (df_processed['womens'] == 1), 'gender_preference'] = 'both'
        
        # 4. Логарифмирование skewed признаков
        df_processed['history_log'] = np.log1p(df_processed['history'])
        
        # 5. Комбинированные признаки для лучшей персонализации
        df_processed['newbie_active'] = (df_processed['newbie'] == 1) & (df_processed['recency'] <= 3)
        df_processed['high_value_active'] = (df_processed['history'] > df_processed['history'].median()) & (df_processed['recency'] <= 3)
        
        return df_processed
    
    def fit_transform(self, df):
        """Предобработка train данных"""
        df_processed = self._create_contextual_features(df)
        
        # Кодирование категориальных переменных
        categorical_cols = ['zip_code', 'channel', 'history_segment', 'value_segment', 
                          'activity_segment', 'gender_preference']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(
                    df_processed[col].astype(str).fillna('unknown')
                )
        
        # Нормализация числовых признаков
        numeric_cols = ['recency', 'history', 'history_log', 'history_recency_ratio', 'customer_value_score']
        numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
        
        if numeric_cols:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        # Определение финального списка признаков
        exclude_cols = ['id', 'segment', 'visit']
        self.feature_columns = [col for col in df_processed.columns if col not in exclude_cols]
        
        print(f"Создано {len(self.feature_columns)} контекстуальных признаков")
        return df_processed
    
    def transform(self, df):
        """Предобработка test данных"""
        df_processed = self._create_contextual_features(df)
        
        # Кодирование категориальных переменных
        categorical_cols = ['zip_code', 'channel', 'history_segment', 'value_segment', 
                          'activity_segment', 'gender_preference']
        
        for col in categorical_cols:
            if col in df_processed.columns and col in self.label_encoders:
                # Обработка новых категорий
                unknown_mask = ~df_processed[col].astype(str).isin(self.label_encoders[col].classes_)
                if unknown_mask.any():
                    most_frequent = self.label_encoders[col].classes_[0]
                    df_processed.loc[unknown_mask, col] = most_frequent
                
                df_processed[col] = self.label_encoders[col].transform(
                    df_processed[col].astype(str).fillna('unknown')
                )
        
        # Нормализация числовых признаков
        numeric_cols = ['recency', 'history', 'history_log', 'history_recency_ratio', 'customer_value_score']
        numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
        
        if numeric_cols and hasattr(self.scaler, 'mean_'):
            available_cols = [col for col in numeric_cols if col in self.scaler.feature_names_in_]
            if available_cols:
                df_processed[available_cols] = self.scaler.transform(df_processed[available_cols])
        
        return df_processed

class EpsilonGreedySGD:
    """Epsilon-Greedy с онлайн обучением через стохастический градиентный спуск"""
    def __init__(self, n_actions=3, n_features=10, epsilon=0.1, learning_rate=0.01, 
                 use_ips=True, decay_epsilon=True):
        self.n_actions = n_actions
        self.n_features = n_features
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.learning_rate = learning_rate
        self.use_ips = use_ips
        self.decay_epsilon = decay_epsilon
        
        # Инициализация весов для каждого действия
        self.weights = np.zeros((n_actions, n_features), dtype=np.float64)
        self.bias = np.zeros(n_actions, dtype=np.float64)
        
        # Статистика для каждого действия
        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)
        self.propensity_calculator = None
        
        print(f"EpsilonGreedySGD инициализирован: {n_actions} действий, {n_features} признаков")
    
    def fit(self, X, y, actions, logging_propensities=None):
        """Онлайн обучение через SGD с IPS коррекцией"""
        n_samples = X.shape[0]
        
        if self.use_ips and logging_propensities is None:
            # Обучаем пропенсити модель если нужно
            self.propensity_calculator = PropensityCalculator()
            self.propensity_calculator.fit(X, actions)
            logging_propensities = self.propensity_calculator.predict_propensity(X, actions)
        
        # Онлайн обучение по каждому примеру
        for i in range(n_samples):
            context = X[i].astype(np.float64)
            action = int(actions[i])
            reward = float(y[i])
            
            # Вычисляем IPS вес если нужно
            if self.use_ips and logging_propensities is not None:
                ips_weight = float(1.0 / logging_propensities[i])
                ips_weight = min(ips_weight, 10.0)  # Клиппинг для контроля дисперсии
            else:
                ips_weight = 1.0
            
            # Обновляем статистику
            self.action_counts[action] += 1
            self.action_rewards[action] += reward
            
            # SGD обновление для выбранного действия
            prediction = self._sigmoid(np.dot(self.weights[action], context) + self.bias[action])
            error = reward - prediction
            
            # Градиентный шаг с IPS весом
            gradient = error * context * ips_weight
            self.weights[action] += self.learning_rate * gradient.astype(np.float64)
            self.bias[action] += self.learning_rate * error * ips_weight
            
            # Decay epsilon для уменьшения exploration со временем
            if self.decay_epsilon:
                total_samples = np.sum(self.action_counts)
                self.epsilon = self.initial_epsilon / (1 + 0.001 * total_samples)
    
    def _sigmoid(self, x):
        """Сигмоида для предсказания вероятности"""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def _compute_q_values(self, context):
        """Вычисление Q-значений для всех действий"""
        q_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            q_values[action] = self._sigmoid(np.dot(self.weights[action], context) + self.bias[action])
        return q_values
    
    def predict_proba_all_actions(self, X):
        """Предсказание политики для всех действий"""
        n_samples = X.shape[0]
        policy_probs = np.zeros((n_samples, self.n_actions))
        
        for i in range(n_samples):
            context = X[i].astype(np.float64)
            q_values = self._compute_q_values(context)
            
            # Epsilon-greedy политика
            best_action = np.argmax(q_values)
            probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
            probs[best_action] += (1 - self.epsilon)
            
            policy_probs[i] = probs
        
        return policy_probs
    
    def get_action_stats(self):
        """Статистика по действиям"""
        stats = {}
        action_names = ['Womens E-Mail', 'Mens E-Mail', 'No E-Mail']
        for action in range(self.n_actions):
            if self.action_counts[action] > 0:
                stats[action_names[action]] = {
                    'count': self.action_counts[action],
                    'total_reward': self.action_rewards[action],
                    'avg_reward': self.action_rewards[action] / self.action_counts[action]
                }
        return stats

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission
    import os
    import pandas as pd
    
    # Загрузка test данных для получения id
    test_df = pd.read_csv('data/test.csv')
    
    submission = test_df[['id']].copy()
    submission['p_mens_email'] = predictions[:, 1]
    submission['p_womens_email'] = predictions[:, 0]  
    submission['p_no_email'] = predictions[:, 2]
    
    # Проверка корректности и нормализация
    prob_sums = submission[['p_mens_email', 'p_womens_email', 'p_no_email']].sum(axis=1)
    
    print(f"Проверка корректности:")
    print(f"  Размер сабмита: {submission.shape}")
    print(f"  Суммы вероятностей: min={prob_sums.min():.8f}, max={prob_sums.max():.8f}")
    
    if abs(prob_sums.mean() - 1.0) > 1e-10:
        print("  Применяем нормализацию...")
        prob_matrix = submission[['p_mens_email', 'p_womens_email', 'p_no_email']].values
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        submission[['p_mens_email', 'p_womens_email', 'p_no_email']] = prob_matrix
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path

def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Загрузка данных
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Предобработка данных
    preprocessor = AdvancedDataPreprocessor()
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)
    
    # Подготовка данных для обучения
    X = train_processed[preprocessor.feature_columns].values
    y = train_processed['visit'].values
    actions = train_processed['segment'].map({
        'Womens E-Mail': 0,
        'Mens E-Mail': 1, 
        'No E-Mail': 2
    }).values
    
    # Обучение EpsilonGreedySGD с IPS коррекцией
    model = EpsilonGreedySGD(
        n_actions=3,
        n_features=X.shape[1],
        epsilon=0.1,
        learning_rate=0.05,
        use_ips=True,
        decay_epsilon=True
    )
    
    # Обучение с IPS коррекцией
    propensity_calc = PropensityCalculator()
    propensity_calc.fit(X, actions)
    logging_propensities = propensity_calc.predict_propensity(X, actions)
    model.fit(X, y, actions, logging_propensities)
    
    # Вывод статистики
    stats = model.get_action_stats()
    print("Статистика действий:")
    for action_name, stat in stats.items():
        print(f"  {action_name}: count={stat['count']:.0f}, avg_reward={stat['avg_reward']:.4f}")
    
    # Создание предсказаний
    X_test = test_processed[preprocessor.feature_columns].values
    predictions = model.predict_proba_all_actions(X_test)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)

if __name__ == "__main__":
    main()