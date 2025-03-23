import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "PCOS_DATASET_AUGMENTED_WITH_BMI.csv"
df = pd.read_csv(file_path)

# Select features and target variable
features = ['Age (yrs)', 'BMI', 'Cycle(R/I)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'TSH (mIU/L)',
            'Follicle No. (L)', 'Follicle No. (R)']
target = 'PCOS (Y/N)'

# Clean and preprocess data
df = df[features + [target]].dropna()
X = df[features].values
y = df[target].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom PCOS detection environment
class PCOSDetectionEnv(gym.Env):
    def __init__(self, X, y):
        super(PCOSDetectionEnv, self).__init__()
        self.X = X
        self.y = y
        self.index = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: No PCOS, 1: PCOS

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        correct = int(action == self.y[self.index])
        reward = 1 if correct else -1
        self.index += 1
        done = self.index >= len(self.X)
        return (self.X[self.index] if not done else np.zeros(self.X.shape[1]), reward, done, {})

# Initialize environment
env = PCOSDetectionEnv(X_train, y_train)

# Q-learning setup
q_table = np.zeros((len(X_train), 2))
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation balance

def train_q_learning(env, q_table, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state_idx = env.index
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_idx])
            
            next_state, reward, done, _ = env.step(action)
            next_state_idx = env.index if not done else state_idx
            q_table[state_idx, action] = (q_table[state_idx, action] +
                                          alpha * (reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]))
    print("Training completed.")

# Train the model
train_q_learning(env, q_table)

def load_model():
    import numpy as np
    import pickle
    
    # Load Q-table and scaler
    q_table = np.load("q_table.npy")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    return q_table, scaler

def predict_pcos(features):
    features = scaler.transform([features])
    best_action = np.argmax(q_table[0])  # Predict using trained Q-table
    return best_action

# Save model
np.save("q_table.npy", q_table)
pd.to_pickle(scaler, "scaler.pkl")
